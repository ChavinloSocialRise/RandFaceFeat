FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app
COPY .. /app

RUN apt-get update
RUN apt-get install -y python3-dev g++ libgl1 git wget
RUN git submodule update --init --recursive

RUN pip install ultralytics==8.2.87 transformers==4.44.2 sentencepiece==0.2.0 accelerate==0.34.0 protobuf==5.28.0 diffusers==0.30.2 torchsde runpod

RUN pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu121

RUN wget https://github.com/ignaciohrdz/yolo-face-parts-detector/releases/download/v2.0.0/yolov8m.pt -O yolov8m.pt

CMD ["python", "-u", "swap/handler.py"]
