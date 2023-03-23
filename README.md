`--vis` option is used for colorful output

Read images from folder:
```
python track_ros.py --yolo-weights yolov8n-seg.pt --images-folder /images/folder [--vis]
```

Read images from topic:
```
python track_ros.py --yolo-weights yolov8n-seg.pt --input-topic /input --output-topic /output [--vis]
```

Additional options:
- `--classes` - consider only these classes (example `--classes 0 1 2`)
