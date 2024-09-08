FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app
COPY .. /app

RUN apt-get update
RUN apt-get install -y python3-dev g++ libgl1 git wget libglib2.0-0
RUN git submodule update --init --recursive

RUN pip install transformers==4.44.2 sentencepiece==0.2.0 accelerate==0.34.0 protobuf==5.28.0 diffusers==0.30.2 torchsde runpod

RUN pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu121

RUN touch inswapper/__init__.py
RUN pip install -r inswapper/requirements.txt

RUN wget https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx -O inswapper/inswapper_128.onnx

CMD ["python", "-u", "multigen/handler.py"]
