FROM python:3.7.9-buster

RUN apt-get update && apt-get install -y libgl1-mesa-glx -y 
# RUN ln -sfn /usr/bin/python3.6 /usr/bin/python3 && ln -sfn /usr/bin/python3 /usr/bin/python && ln -sfn /usr/bin/pip3 /usr/bin/pip

COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
RUN pip install .

# CMD ["python", "server.py"]

EXPOSE 27004
