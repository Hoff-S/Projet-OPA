FROM ubuntu:22.04
ADD requirements.txt fastapi_main.py ./
RUN apt-get update && apt-get install python3-pip -y && pip3 install -r requirements.txt

EXPOSE 8000
CMD uvicorn fastapi_main:api --host 0.0.0.0
