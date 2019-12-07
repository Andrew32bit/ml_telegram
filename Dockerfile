FROM python:3.6.9-stretch
RUN apt-get update && apt-get install -y tdsodbc unixodbc-dev && apt-get install -y libsasl2-dev gcc python-dev libsasl2-2 libsasl2-modules-gssapi-mit \
 && apt install unixodbc-bin -y  \
 && apt-get clean -y
MAINTAINER ANDREW KONSTANTINOV
COPY requirements.txt /add
COPY app.py /add
COPY dict.xlsx /add
COPY all.norm-sz100-w10-cb0-it1-min100.w2v /add
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["python3","app.py"]

