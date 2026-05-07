#!/bin/bash

# Network
if [ -z ${HSR_IP} ]; then
    if [ ! -z ${ROBOT_NAME} ]; then
        HSR_NAME=$ROBOT_NAME
    else
        echo "Set name of ROBOT_NAME or HSR_IP (e.g. 'export ROBOT_NAME=hsrc / export HSR_IP=192.168.0.100')"
        exit 1
    fi

    # Configure the known host names with '/etc/hosts' in the Docker container.
    HSR_HOSTNAME=${HSR_NAME}.local
    echo "Now resolving local host name '${HSR_HOSTNAME}'..."
    HSR_IP=`avahi-resolve -4 --name ${HSR_HOSTNAME} | cut -f 2`
    if [ "$?" != "0" ]; then
        echo "Failed to execute 'avahi-resolve'. You may need to install 'avahi-utils'."
        exit 1
    elif [ ! -z "${HSR_IP}" ]; then
        echo "Successfully resolved host name '${HSR_HOSTNAME}' as '${HSR_IP}'."
    else
        echo "Failed to resolve host name '${HSR_HOSTNAME}'."
        exit 1
    fi
else
    HSR_IP=${HSR_IP}
fi

# Dataset
if [ -n "$HSR_DATASET_DIR" ]; then
  echo "HSR_DATASET_DIR is set to '$HSR_DATASET_DIR'. Mounting this path to the container as '/root/datasets'."
else
  export HSR_DATASET_DIR="$PWD/datasets"
  echo "HSR_DATASET_DIR is not set. Using default path: $HSR_DATASET_DIR"
fi

TAG_NAME=noetic-cuda11
IMAGE_NAME=hsr_data_collection
CONTAINER_NAME=hsr_data_collection_${USER}
ROS_IP=`hostname -I | cut -d' ' -f1`
echo "$0: IMAGE=${IMAGE_NAME}"
echo "$0: CONTAINER=${CONTAINER_NAME}"
echo "$0: HSR_IP=${HSR_IP}"
echo "$0: ROS_IP=${ROS_IP}"

GIT_HASH=$(git rev-parse HEAD)
GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD)

xhost +

EXISTING_CONTAINER_ID=`docker ps -aq -f name=${CONTAINER_NAME}`
if [ ! -z "${EXISTING_CONTAINER_ID}" ]; then
    RUNNING_CONTAINER_ID=`docker ps -aq -f name=${CONTAINER_NAME} -f status=running`
    if [ -z "${RUNNING_CONTAINER_ID}" ]; then
        docker container start ${CONTAINER_NAME}
    fi
    docker exec -it ${CONTAINER_NAME} bash
else
    docker run -it \
        --privileged \
        --gpus all \
        --net host \
        --env DISPLAY=${DISPLAY} \
        --env GIT_HASH=${GIT_HASH} \
        --env GIT_BRANCH=${GIT_BRANCH} \
        --volume ${PWD}/conversion/:/root/conversion/ \
        --volume ${PWD}/hsr_data_collection/:/root/catkin_ws/src/hsr_data_collection/ \
        --volume ${PWD}/hsr_data_msgs/:/root/catkin_ws/src/hsr_data_msgs/ \
        --volume ${PWD}/hsrb_pseudo_endeffector_position_controller/:/root/catkin_ws/src/hsrb_pseudo_endeffector_position_controller/ \
        --volume ${HSR_DATASET_DIR}/:/root/datasets/ \
        --volume /dev/:/dev/ \
        --volume /tmp/.X11-unix:/tmp/.X11-unix \
        --name ${CONTAINER_NAME} \
        ${IMAGE_NAME}:${TAG_NAME} \
        bash -c "sed -i 's/TMP_IP/${ROS_IP}/' ~/.bashrc;
                 sed -i 's/localhost:11311/${HSR_IP}:11311/' ~/.bashrc;
                 source ~/.bashrc;
                 bash"
fi
