FROM nvidia/cudagl:11.3.0-devel-ubuntu20.04 as base

ARG HSR_APT_USER=
ARG HSR_APT_PASSWORD=
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV OPENPI_PROJECT_PYTHON=3.11

SHELL ["/bin/bash", "-c"]

# This legacy ROS data-collection image keeps Ubuntu 20.04's system Python and is not the primary OpenPI training runtime.

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        wget \
        curl \
        sudo \
        software-properties-common \
        lsb-release && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# ROS
## https://wiki.ros.org/noetic/Installation/Ubuntu
### Setup source.list
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
### Setup keys
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
###  Install ros-noetic-desktop-full
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        ros-noetic-desktop-full && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
###  Dependencies for building packages
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        python3-pip \
        python3-rosdep \
        python3-rosinstall \
        python3-rosinstall-generator \
        python3-wstool \
        python3-catkin-tools \
        build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    rosdep init && \
    rosdep update

### Install ros-noetic-tmc-desktop-full
RUN if [ -z "${HSR_APT_USER}" ] || [ -z "${HSR_APT_PASSWORD}" ]; then \
      echo "HSR_APT_USER and HSR_APT_PASSWORD are required to build data collection image." >&2; \
      exit 1; \
    fi
RUN printf 'deb [arch=amd64] https://%s:%s@packages.hsr.io/ros/ubuntu %s main\\n' "${HSR_APT_USER}" "${HSR_APT_PASSWORD}" "$(lsb_release -cs)" > /etc/apt/sources.list.d/tmc.list && printf 'deb [arch=amd64] https://%s:%s@packages.hsr.io/tmc/ubuntu %s multiverse main\\n' "${HSR_APT_USER}" "${HSR_APT_PASSWORD}" "$(lsb_release -cs)" >> /etc/apt/sources.list.d/tmc.list
RUN sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-stable.list'
RUN wget https://${HSR_APT_USER}:${HSR_APT_PASSWORD}@packages.hsr.io/tmc.key -O - | apt-key add -
RUN wget https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc -O - | apt-key add -
RUN wget https://packages.osrfoundation.org/gazebo.key -O - | apt-key add -
RUN install -d /etc/apt/auth.conf.d && printf 'machine packages.hsr.io\\nlogin %s\\npassword %s\\n' "${HSR_APT_USER}" "${HSR_APT_PASSWORD}" > /etc/apt/auth.conf.d/auth.conf && chmod 600 /etc/apt/auth.conf.d/auth.conf
RUN sh -c '/bin/echo -e "Package: ros-noetic-laser-ortho-projector\nPin: version 0.3.3*\nPin-Priority: 1001\n\nPackage: ros-noetic-laser-scan-matcher\nPin: version 0.3.3*\nPin-Priority: 1001\n\nPackage: ros-noetic-laser-scan-sparsifier\nPin: version 0.3.3*\nPin-Priority: 1001\n\nPackage: ros-noetic-laser-scan-splitter\nPin: version 0.3.3*\nPin-Priority: 1001\n\nPackage: ros-noetic-ncd-parser\nPin: version 0.3.3*\nPin-Priority: 1001\n\nPackage: ros-noetic-polar-scan-matcher\nPin: version 0.3.3*\nPin-Priority: 1001\n\nPackage: ros-noetic-scan-to-cloud-converter\nPin: version 0.3.3*\nPin-Priority: 1001\n\nPackage: ros-noetic-scan-tools\nPin: version 0.3.3*\nPin-Priority: 1001" > /etc/apt/preferences'
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        ros-noetic-tmc-desktop-full && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN echo "export ROS_HOME=${HOME}/.ros" >> /root/.bashrc
RUN echo 'export ROS_MASTER_URI="${ROS_MASTER_URI:-http://localhost:11311}"' >> /root/.bashrc
RUN echo 'export ROS_IP="${ROS_IP:-}"' >> /root/.bashrc
RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc

CMD ["/bin/bash"]


FROM base as hsrtx

ARG HSR_APT_USER=
ARG HSR_APT_PASSWORD=
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV OPENPI_PROJECT_PYTHON=3.11

# Install external Libraries
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        spacenavd \
        ros-noetic-spacenav-node \
        git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# external ROS setting
WORKDIR /root
RUN mkdir -p external_catkin_ws/src && \
    cd /root/external_catkin_ws/src && \
    git clone https://github.com/IFL-CAMP/tf_bag.git && \
    cd /root/external_catkin_ws && \
    bash -c "source /opt/ros/noetic/setup.bash; catkin build"

RUN pip3 install --no-cache-dir \
        scipy==1.10.1

RUN pip3 install -U --no-cache-dir \
        numpy \
        matplotlib \
        pandas \
        rlds[tensorflow] \
        tfds-nightly \
        albumentations

RUN echo 'export ROS_PACKAGE_PATH="${ROS_PACKAGE_PATH:+${ROS_PACKAGE_PATH}:}/root/external_catkin_ws/src/"' >> /root/.bashrc
RUN echo 'if [ -f /root/external_catkin_ws/devel/setup.bash ]; then source /root/external_catkin_ws/devel/setup.bash; fi' >> /root/.bashrc

# ROS workspace
WORKDIR /root
RUN mkdir -p catkin_ws/src && \
    cd /root/catkin_ws && \
    bash -c "source /opt/ros/noetic/setup.bash; catkin build"

RUN echo 'export ROS_PACKAGE_PATH="${ROS_PACKAGE_PATH:+${ROS_PACKAGE_PATH}:}/root/catkin_ws/src/"' >> /root/.bashrc
RUN echo "export ROS_WORKSPACE=/root/catkin_ws" >> /root/.bashrc
RUN echo 'if [ -f /root/catkin_ws/devel/setup.bash ]; then source /root/catkin_ws/devel/setup.bash; fi' >> /root/.bashrc

ENTRYPOINT []
CMD ["/bin/bash"]
