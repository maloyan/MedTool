FROM nvcr.io/nvidia/pytorch:21.05-py3

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt

RUN pip install -r requirements.txt