FROM ubuntu:22.04

ADD requirements.txt model_training.py ./

RUN apt-get update && apt-get install python3-pip -y && pip3 install -r requirements.txt

CMD python3 model_training.py