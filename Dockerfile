FROM nvcr.io/nvidia/pytorch:21.03-py3

ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y zip htop screen libgl1-mesa-glx ffmpeg

COPY dev_requirements.txt .
RUN python -m pip install --upgrade pip
RUN pip uninstall -y nvidia-tensorboard nvidia-tensorboard-plugin-dlprof
RUN pip install --no-cache coremltools onnx gsutil notebook

RUN pip install --no-cache \
  PyYAML==5.4.1 \
  tensorboard==2.4.1 \
  seaborn==0.11.1 \
  coremltools==4.1 \
  onnx>=1.8.0 \
  onnxruntime==1.8.0 \
  numpy==1.19.0 \
  thop==0.0.31-2005241907 \
  pycocotools==2.0.2

RUN pip install --no-cache supervisely==6.73.522

# np.object were removed in NumPy 2.x. Force NumPy < 2.0 to avoid:
# "AttributeError: module 'numpy' has no attribute 'object'"
RUN pip install --no-cache --force-reinstall --no-deps "numpy==1.23.5"

# Remove opencv-python version installed by supervisely
RUN pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless || true

# keep opencv-python==4.6.0.66 installed to satisfy supervisely's dependency check,
# but remove supervisely cv2 module and provide a stable cv2 implementation via
# opencv-python-headless==4.5.5.62
RUN pip install --no-cache --force-reinstall --no-deps opencv-python==4.6.0.66 \
 && rm -rf /opt/conda/lib/python3.8/site-packages/cv2 \
 && pip install --no-cache --force-reinstall --no-deps opencv-python-headless==4.5.5.62 \
 && python -c "import pkg_resources; print('opencv-python dist:', pkg_resources.get_distribution('opencv-python').version)" \
 && python -c "import cv2; print('cv2 import OK, version:', cv2.__version__)"

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app
COPY . /usr/src/app
ENV HOME=/usr/src/app

LABEL python_sdk_version="6.73.522"
