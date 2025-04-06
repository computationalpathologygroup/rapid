# =============================================================
# Configuration
# =============================================================
ARG UBUNTU_VERSION=20.04
ARG CUDA_MAJOR_VERSION=11.3.1
ARG CUDNN_MAJOR_VERSION=8
ARG PYTHON_VERSION=3.9.15
ARG NUMPY_VERSION=1.21.5
ARG VIPS_VERSION=8.13.0

ARG BUILD_JOBS=16

# =============================================================
# Create build docker
# =============================================================

FROM nvidia/cuda:${CUDA_MAJOR_VERSION}-cudnn${CUDNN_MAJOR_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS builder

# === Propagate build args ===
ARG PYTHON_VERSION
ARG NUMPY_VERSION
ARG BUILD_JOBS

# === Install build packages ===
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        pkg-config apt-transport-https \
        openjdk-8-jdk \
        g++ ninja-build make \
        wget git zip unzip \
        libssl-dev zlib1g-dev \
        libncurses5-dev libncursesw5-dev libreadline-dev libsqlite3-dev \
        libgdbm-dev libdb5.3-dev libbz2-dev libexpat1-dev liblzma-dev tk-dev \
        gcovr libffi-dev uuid-dev

# === Install python ===
RUN cd /tmp && \
    wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tar.xz && \
    tar xfv Python*.xz && \
    cd Python-3*/ && \
    ./configure --enable-shared LDFLAGS="-fprofile-arcs" && \
    make -j${BUILD_JOBS} install && \
    cd /tmp && \
    rm -rf Python-3* && \
    ldconfig

RUN pip3 install --upgrade pip && \
    pip3 install cython wheel numpy==${NUMPY_VERSION} packaging

# =============================================================
# Create base docker
# =============================================================

FROM nvidia/cuda:${CUDA_MAJOR_VERSION}-cudnn${CUDNN_MAJOR_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS base

# === Propagate build args ===
ARG CUDA_MAJOR_VERSION
ARG PYTHON_VERSION
ARG NUMPY_VERSION
ARG BUILD_JOBS

# === Configurate environment variables ===
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV DEBIAN_FRONTEND noninteractive

# === Install libs ===
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
        man \
        openssh-server \
        ca-certificates apt-transport-https \
        sudo \
        git subversion \
        nano vim \
        tmux screen \
        htop \
        g++ ninja-build \
        rsync \
        pv \
        meson \
        curl wget \
        bzip2 zip unzip \
        dcmtk libboost-all-dev \
        libgomp1 \
        libjpeg-turbo8 \
        libssl-dev zlib1g-dev libncurses5-dev libncursesw5-dev libreadline-dev libsqlite3-dev \
        libgdbm-dev libdb5.3-dev libbz2-dev libexpat1-dev liblzma-dev tk-dev gcovr libffi-dev uuid-dev \
        libgtk2.0-dev libgsf-1-dev libtiff5-dev libopenslide-dev \
        libgl1-mesa-glx && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir /var/run/sshd && \
    cat /etc/sudoers | grep -v secure_path > /tmp/sudoers && mv /tmp/sudoers /etc/sudoers

# == Install VIPS  ==
ARG VIPS_VERSION

RUN wget https://github.com/libvips/libvips/releases/download/v${VIPS_VERSION}/vips-${VIPS_VERSION}.tar.gz -P /tmp && \
    tar -xf /tmp/vips-${VIPS_VERSION}.tar.gz --directory /tmp/ && \
    cd /tmp/vips-${VIPS_VERSION} && \
    ./configure && \
    make && \
    sudo -S make install && \
    cd .. && \
    sudo -S ldconfig

RUN env | grep '^PATH=\|^LD_LIBRARY_PATH=\|^LANG=\|^LC_ALL=\|^CUDA_ROOT=' > /etc/environment

# === Configure timezone ===
RUN echo "Europe/Amsterdam" > /etc/timezone && \
    rm -f /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata

# === Setup user ===
RUN useradd -ms /bin/bash user && \
    (echo user ; echo user) | passwd user && \
    gpasswd -a user ssh && \
    gpasswd -a user sudo

# === Install python with up-to-date pip ===
RUN cd /tmp && \
    wget "https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tar.xz" && \
    tar xfv Python*.xz && \
    cd Python-3*/ && \
    ./configure --enable-shared LDFLAGS="-fprofile-arcs" && \
    make -j${BUILD_JOBS} install && \
    cd ~ && \
    rm -rf /tmp/Python-3* && \
    ldconfig

RUN pip3 install --upgrade pip pip-tools wheel setuptools && \
    printf '#!/bin/bash\necho "Please use pip3 instead of pip to install packages for python3"' > /usr/local/bin/pip && \
    chmod +x /usr/local/bin/pip && \
    rm -rf ~/.cache/pip

# === Install python libraries ===
RUN pip3 install --upgrade pip setuptools wheel
COPY base_requirements.txt /root/base_requirements.txt
RUN pip3 install -r /root/base_requirements.txt
COPY additional_requirements.txt /root/additional_requirements.txt
RUN pip3 install -r /root/additional_requirements.txt

# === Install ASAP ===
RUN wget https://github.com/computationalpathologygroup/ASAP/releases/download/ASAP-2.1/ASAP-2.1-py39-Ubuntu2004.deb -P /root/
RUN apt-get update && \
    apt-get install --assume-yes /root/ASAP-2.1-py39-Ubuntu2004.deb && \
    ldconfig && \
    SITE_PACKAGES=`python3 -c "import sysconfig; print(sysconfig.get_paths()['purelib'])"` && \
    printf "/opt/ASAP/bin/\n" > "${SITE_PACKAGES}/asap.pth" && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# === Install latest Pixman ===
#  This should fix the bug described here (https://github.com/TissueImageAnalytics/tiatoolbox/issues/133)
RUN sudo apt-get install meson && \
    wget https://www.cairographics.org/releases/pixman-0.42.2.tar.gz && \
    tar -xf pixman-0.42.2.tar.gz && \
    cd pixman-0.42.2 && \
    mkdir build && \
    cd build && \
    meson --prefix=/usr --buildtype=release && \
    ninja && \
    sudo ninja install

# === Development environment setup ===
RUN echo "export PATH=/usr/local/bin:$PATH" >> /home/user/.bashrc && \
    echo "export PYTHONPATH=/workspace:$PYTHONPATH" >> /home/user/.bashrc && \
    chown -R user:user /home/user

# === Install different feature detectors ===
RUN mkdir /detectors 
WORKDIR /detectors

# DALF
RUN git clone https://github.com/verlab/DALF_CVPR_2023.git

# LightGLUE
RUN git clone https://github.com/cvg/LightGlue.git && \
  cd LightGlue && \
  python3 -m pip install -e . && \
  cd ..

# RoMa
RUN git clone https://github.com/dnschouten/RoMa.git && \
 cd RoMa && \
 pip3 install -e . && \
 cd ..

# DeDoDe
RUN git clone https://github.com/Parskatt/DeDoDe.git && \
  cd DeDoDe && \
  pip3 install -e . && \
  cd assets && \
  wget https://github.com/Parskatt/DeDoDe/releases/download/dedode_pretrained_models/dedode_descriptor_B.pth && \
  wget https://github.com/Parskatt/DeDoDe/releases/download/dedode_pretrained_models/dedode_descriptor_G.pth && \
  wget https://github.com/Parskatt/DeDoDe/releases/download/dedode_pretrained_models/dedode_detector_L.pth && \
  wget https://github.com/Parskatt/DeDoDe/releases/download/v2/dedode_detector_L_v2.pth && \
  cd ../..

# Ugly hack to get right torch + cuda version for all of the new detectors
RUN pip3 install torchvision==0.15.2+cu117 -f https://download.pytorch.org/whl/torch_stable.html

# === Expose ports for ssh, tensorboard, and jupyter. ===
EXPOSE 22 6006 8888

# === Overwrite NVIDIA's authorship label
LABEL maintainer="DNSCHOUTEN"

# Configure entrypoint
USER root
COPY run.sh /home/user/run.sh
ENTRYPOINT ["/bin/bash", "/home/user/run.sh"]