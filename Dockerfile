FROM ubuntu:20.04

LABEL key="maloyan"
RUN apt-get update && apt-get install -y \
    python3-pip

WORKDIR /workspace

COPY . /workspace/

RUN pip install -r requirements.txt
RUN pip install -e .