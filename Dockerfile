FROM nvcr.io/nvidia/pytorch:20.07-py3
COPY requirements.txt .

RUN pip install -r requirements.txt
