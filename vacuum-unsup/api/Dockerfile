FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

# install netcat
RUN apt-get update -y && \
    apt-get -y install netcat && \
    apt-get clean && \
    apt install libgl1-mesa-glx -y

RUN apt-get install 'ffmpeg'\
    'libsm6'\
    'libxext6'  -y

RUN pip3 install --upgrade pip

# set working directory
WORKDIR /usr/src/app

# add and install requirements
COPY ./requirements.txt /usr/src/app/requirements.txt
RUN pip3 install -r requirements.txt

# add app
COPY . /usr/src/app

# image classification service by twistcode
EXPOSE 5001

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]