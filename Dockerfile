FROM nvcr.io/nvidia/pytorch:21.03-py3

ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y zip htop screen libgl1-mesa-glx ffmpeg

COPY dev_requirements.txt .
RUN python -m pip install --upgrade pip
RUN pip uninstall -y nvidia-tensorboard nvidia-tensorboard-plugin-dlprof
RUN pip install --no-cache coremltools onnx gsutil notebook

RUN pip install --no-cache \
  opencv-python-headless==4.5.5.62 \
  opencv-python==4.5.5.62 \
  PyYAML==5.4.1 \
  tensorboard==2.4.1 \
  seaborn==0.11.1 \
  coremltools==4.1 \
  onnx>=1.8.0 \
  onnxruntime==1.8.0 \
  numpy==1.19.0 \
  thop==0.0.31-2005241907 \
  pycocotools==2.0.2

RUN pip install --no-cache supervisely==6.73.486

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app
COPY . /usr/src/app
ENV HOME=/usr/src/app

LABEL python_sdk_version="6.73.486"
