FROM nvcr.io/nvidia/pytorch:21.05-py3

WORKDIR /workspace

COPY requirement.txt /workspace/requirement.txt

RUN pip install -r requirement.txt