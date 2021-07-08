FROM python:3.8 AS builder
COPY requirements.txt .

RUN pip install -r requirements.txt
