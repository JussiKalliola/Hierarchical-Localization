FROM colmap/colmap:latest
MAINTAINER Paul-Edouard Sarlin
ARG PYTHON_VERSION=3.8
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-get update -y
RUN apt-get install -y unzip wget software-properties-common tmux && python3-tk
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get -y update && \
    apt-get install -y python${PYTHON_VERSION}
RUN wget https://bootstrap.pypa.io/get-pip.py && python${PYTHON_VERSION} get-pip.py
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1
COPY . /app
WORKDIR app/
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
RUN pip3 install notebook
