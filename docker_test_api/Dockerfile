FROM ubuntu:22.04
RUN apt-get update && apt-get install python3-pip -y
ADD requirements.txt ./
RUN pip3 install -r requirements.txt
ADD test_fastapi_main.py ./
CMD python3 test_fastapi_main.py
