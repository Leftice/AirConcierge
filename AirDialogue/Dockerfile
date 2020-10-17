FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu16.04

ENV LD_LIBRARY_PATH /usr/local/cuda-10.0/lib64:/usr/local/cuda-10.0/extras/CUPTI/lib64:$LD_LIBRARY_PATH

RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git libgtk2.0-dev locales && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install python 2.7
RUN apt-get update && \
    apt-get install -y python python-pip && \
    apt-get install -y python-tk && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install python 3.5
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set locale
RUN locale-gen en_US.UTF-8  
ENV LANG en_US.UTF-8  
ENV LANGUAGE en_US:en  
ENV LC_ALL en_US.UTF-8

# Install packages in python 2.7
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --upgrade setuptools && \
    pip install --no-cache-dir numpy==1.16.6 Pillow==6.2.2 PyYAML==5.3.1 six==1.15.0 wheel==0.35.1 && \
    pip install --no-cache-dir sklearn matplotlib dill tensorboardX scipy pandas sympy nose tqdm scikit-learn scikit-image nltk && \
    pip install --no-cache-dir torch==1.3.1+cu100 torchvision==0.4.2+cu100 -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install --no-cache-dir torchtext==0.4.0 
    pip install --no-cache-dir tensorflow-gpu==1.14.0

# Install packages in python 3.5
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir --upgrade setuptools && \
    pip3 install --no-cache-dir numpy==1.16.6 Pillow==6.2.2 PyYAML==5.3.1 six==1.15.0 wheel==0.35.1 && \
    pip3 install --no-cache-dir sklearn matplotlib dill tensorboardX scipy pandas sympy nose tqdm scikit-learn scikit-image nltk && \
    pip3 install --no-cache-dir argparse

# Initialize workspace
WORKDIR /

