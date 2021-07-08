FROM ubuntu:20.04

WORKDIR /workspace

COPY . /workspace/

RUN pip install -r requirements.txt
RUN pip install -e .
