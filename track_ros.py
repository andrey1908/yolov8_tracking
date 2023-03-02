import argparse
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image as RosImage
import cv2
import os
# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import platform
import numpy as np
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov8') not in sys.path:
    sys.path.append(str(ROOT / 'yolov8'))  # add yolov5 ROOT to PATH
if str(ROOT / 'trackers' / 'strongsort') not in sys.path:
    sys.path.append(str(ROOT / 'trackers' / 'strongsort'))  # add strong_sort ROOT to PATH

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import logging
from yolov8.ultralytics.nn.autobackend import AutoBackend
from yolov8.ultralytics.yolo.data.dataloaders.stream_loaders import LoadImages, LoadStreams, RosImages
from yolov8.ultralytics.yolo.data.utils import IMG_FORMATS, VID_FORMATS
from yolov8.ultralytics.yolo.utils import DEFAULT_CFG, LOGGER, SETTINGS, callbacks, colorstr, ops
from yolov8.ultralytics.yolo.utils.checks import check_file, check_imgsz, check_imshow, print_args, check_requirements
from yolov8.ultralytics.yolo.utils.files import increment_path
from yolov8.ultralytics.yolo.utils.torch_utils import select_device
from yolov8.ultralytics.yolo.utils.ops import Profile, non_max_suppression, scale_boxes, process_mask, process_mask_native
from yolov8.ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box

from trackers.multi_tracker_zoo import create_tracker


class RosTracker:
    def __init__(
            self,
            source='0',
            yolo_weights=WEIGHTS / 'yolov5m.pt',  # model.pt path(s),
            reid_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt',  # model.pt path,
            tracking_method='strongsort',
            tracking_config=None,
            imgsz=(640, 640),  # inference size (height, width)
            conf_thres=0.25,  # confidence threshold
            iou_thres=0.45,  # NMS IOU threshold
            max_det=1000,  # maximum detections per image
            device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            show_vid=False,  # show results
            save_txt=False,  # save results to *.txt
            save_conf=False,  # save confidences in --save-txt labels
            save_crop=False,  # save cropped prediction boxes
            save_trajectories=False,  # save trajectories for each track
            save_vid=False,  # save confidences in --save-txt labels
            nosave=False,  # do not save images/videos
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            visualize=False,  # visualize features
            update=False,  # update all models
            project=ROOT / 'runs' / 'track',  # save results to project/name
            name='exp',  # save results to project/name
            exist_ok=False,  # existing project/name ok, do not increment
            line_thickness=2,  # bounding box thickness (pixels)
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
            hide_class=False,  # hide IDs
            half=False,  # use FP16 half-precision inference
            dnn=False,  # use OpenCV DNN for ONNX inference
            vid_stride=1,  # video frame-rate stride
            retina_masks=False,
            vis=False):

        # Load model
        self.half = half
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.max_det = max_det
        self.vis = vis

        self.device = select_device(device)
        self.model = AutoBackend(
            yolo_weights, device=self.device, dnn=dnn, fp16=self.half)
        self.stride, self.names, self.pt = \
            self.model.stride, self.model.names, self.model.pt
        print(self.names)
        self.imgsz = check_imgsz(imgsz, stride=self.stride)  # check image size

        self.model.warmup(imgsz=(1, 3, *self.imgsz))  # warmup

        # Create as many strong sort instances as there are video sources
        self.tracker = create_tracker(
            tracking_method, tracking_config, reid_weights, self.device, self.half)
        if hasattr(self.tracker, 'model'):
            if hasattr(self.tracker.model, 'warmup'):
                self.tracker.model.warmup()

        # Run tracking
        self.seen, self.dt = \
            0, (Profile(), Profile(), Profile(), Profile())
        self.curr_frame, self.prev_frame = None, None

        # Dataloader
        self.dataset = RosImages(
            "/kitti/camera_color_left/image_raw",
            self.track_callback,
            imgsz=self.imgsz,
            stride=self.stride,
            auto=self.pt,
            transforms=getattr(self.model.model, 'transforms', None))
        self.pub = rospy.Publisher("/tracked_image", RosImage, queue_size=1)
        self.bridge = CvBridge()

    @torch.no_grad()
    def track_callback(self, im, im0, header):
        with self.dt[0]:
            im = torch.from_numpy(im).to(self.device)
            im = im.half() if self.half else im.float()  # uint8 to fp16/32
            im /= 255.0  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with self.dt[1]:
            preds = self.model(im)

        # Apply NMS
        with self.dt[2]:
            p = non_max_suppression(preds[0], self.conf_thres, self.iou_thres,
                self.classes, self.agnostic_nms, max_det=self.max_det, nm=32)
            proto = preds[1][-1]
            
        # Process detections
        det = p[0]
        self.seen += 1
        self.curr_frame = im0
        
        if hasattr(self.tracker, 'tracker') and hasattr(self.tracker.tracker, 'camera_update'):
            if self.prev_frame is not None and self.curr_frame is not None:  # camera motion compensation
                self.tracker.tracker.camera_update(self.prev_frame, self.curr_frame)

        out_img = torch.zeros(im.shape, device=im.device).squeeze().permute(1, 2, 0)
        if det is not None and len(det):
            shape = im0.shape
            masks = process_mask(
                proto[0], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
            det[:, :4] = scale_boxes(
                im.shape[2:], det[:, :4], shape).round()  # rescale boxes to im0 size

            # pass detections to strongsort
            with self.dt[3]:
                output = self.tracker.update(det.cpu(), im0, masks)
            
            # draw boxes for visualization
            for out in output:
                bbox = out[0:4]
                id = out[4]
                cls = out[5]
                conf = out[6]
                mask = out[7]
                if id == 0:
                    rospy.logfatal("id = 0")
                    rospy.signal_shutdown("id = 0")

                # choose color mode
                if self.vis:
                    color = colors(id)
                else:
                    color = torch.zeros(3)
                    color[0] = id & 0xFF
                    color[1] = (id >> 8) & 0xFF
                    color[2] = (id >> 16) & 0xFF
                    color.to(im.device)

                out_img[mask] = torch.Tensor(color).to(im.device)
        else:
            pass
            
        # Stream results
        out_img = out_img.byte().cpu().numpy()
        out_img = cv2.resize(out_img, (im0.shape[1], im0.shape[0]),
            interpolation=cv2.INTER_NEAREST)
        msg = self.bridge.cv2_to_imgmsg(out_img, encoding="bgr8")
        msg.header = header
        self.pub.publish(msg)

        self.prev_frame = self.curr_frame


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', nargs='+', type=Path, default=WEIGHTS / 'yolov8s-seg.pt', help='model.pt path(s)')
    parser.add_argument('--reid-weights', type=Path, default=WEIGHTS / 'osnet_x0_25_msmt17.pt')
    parser.add_argument('--tracking-method', type=str, default='bytetrack', help='strongsort, ocsort, bytetrack')
    parser.add_argument('--tracking-config', type=Path, default=None)
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')  
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--save-trajectories', action='store_true', help='save trajectories for each track')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs' / 'track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--hide-class', default=False, action='store_true', help='hide IDs')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--retina-masks', action='store_true', help='whether to plot masks in native resolution')
    parser.add_argument('--vis', action='store_true')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    opt.tracking_config = ROOT / 'trackers' / opt.tracking_method / 'configs' / (opt.tracking_method + '.yaml')
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    
    rospy.init_node("tracker")
    ros_tracker = RosTracker(**vars(opt))
    rospy.spin()


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
