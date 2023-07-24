FROM nvidia/cuda:11.2.2-devel-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive

ARG NUM_THREADS=1

# User setup
ARG USER=docker_yolov8_tracking
ARG UID=1000
ARG GID=1000
ARG PW=user

# Update image
RUN apt-get update && \
    apt-get install -y \
        python3 \
        python3-pip \
        lsb-release \
        curl

RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' && \
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - && \
    apt-get update && \
    apt-get install -y \
        ros-noetic-ros-base \
        ros-noetic-cv-bridge

# Install pip packages
RUN pip3 install --upgrade pip
RUN pip install --upgrade setuptools wheel
RUN pip install \
        tqdm \
        pandas==1.5.3 \
        numpy==1.23.1 \
        matplotlib==3.7.1 \
        psutil==5.7.3 \
        easydict \
        torch==1.9.0 \
        torchvision==0.10.0 \
        ipython \
        thop \
        sentry-sdk \
        requests \
        ultralytics==8.0.20 \
        gitpython \
        opencv-python==4.6.0.66 \
        Pillow==9.5.0 \
        scipy==1.10.1 \
        seaborn==0.12.2 \
        gdown==4.6.4 \
        lap==0.4.0 \
        filterpy==1.4.5

# Add user and his password
RUN useradd -m ${USER} --uid=${UID} && \
    echo "${USER}:${PW}" | chpasswd && \
    usermod -s /bin/bash ${USER} && \
    usermod -a -G sudo ${USER}

WORKDIR /home/${USER}
USER ${UID}:${GID}

