#!/bin/bash

openpi_require_docker_project() {
  if [ -z "${DEEP_PROJECT_NAME:-}" ]; then
    echo "Set DEEP_PROJECT_NAME (e.g. 'export DEEP_PROJECT_NAME=mytest')"
    exit 1
  fi

  PROJECT="${DEEP_PROJECT_NAME}"
  CONTAINER="${PROJECT}_deep_1"
  export PROJECT CONTAINER
  export HOSTNAME="${HOSTNAME:-$(hostname)}"
}

openpi_set_docker_mode() {
  MODE="${1:-train}"
  case "${MODE}" in
    train)
      export DEEP_DOCKERFILE=./docker/Dockerfile.train
      : "${DEEP_IMAGE_TAG:=train}"
      : "${OPENPI_ENABLE_ROS_INIT:=0}"
      ;;
    train-gb200)
      export DEEP_DOCKERFILE=./docker/Dockerfile.train.gb200
      : "${DEEP_IMAGE_TAG:=train-gb200}"
      : "${OPENPI_ENABLE_ROS_INIT:=0}"
      ;;
    deploy)
      export DEEP_DOCKERFILE=./docker/Dockerfile.deploy
      : "${DEEP_IMAGE_TAG:=deploy}"
      : "${OPENPI_ENABLE_ROS_INIT:=1}"
      ;;
    *)
      echo "Unknown mode: ${MODE} (expected 'train', 'train-gb200', or 'deploy')"
      exit 1
      ;;
  esac

  : "${OPENPI_PROJECT_PYTHON:=3.11}"
  export MODE DEEP_IMAGE_TAG OPENPI_ENABLE_ROS_INIT OPENPI_PROJECT_PYTHON
  export OPENPI_DOCKER_MODE="${MODE}"
}

openpi_prepare_docker_paths() {
  if [ -z "${DEEP_DATASET_PATH:-}" ]; then
    export DEEP_DATASET_PATH="$(pwd)/datasets"
    echo "DEEP_DATASET_PATH is not set. Using ${DEEP_DATASET_PATH}"
  fi

  if [ -z "${DEEP_CACHE_ROOT_HOST:-}" ]; then
    export DEEP_CACHE_ROOT_HOST="$(pwd)/.docker_cache"
    echo "DEEP_CACHE_ROOT_HOST is not set. Using ${DEEP_CACHE_ROOT_HOST}"
  fi

  mkdir -p "${DEEP_DATASET_PATH}"
  mkdir -p "${DEEP_CACHE_ROOT_HOST}"/{uv-cache,uv-data,uv-python,openpi_cache,huggingface_home,tmp,cache}
}

openpi_require_deploy_build_credentials() {
  if [ "${MODE:-}" != "deploy" ]; then
    return
  fi

  if [ -z "${HSR_APT_USER:-}" ] || [ -z "${HSR_APT_PASSWORD:-}" ]; then
    echo "Set HSR_APT_USER and HSR_APT_PASSWORD before building deploy images."
    exit 1
  fi
}

openpi_resolve_deploy_network() {
  if [ "${MODE}" != "deploy" ]; then
    echo "[INFO] Skipping HSR network setup in train mode."
    return
  fi

  if [ -n "${HSR_IP:-}" ]; then
    echo "Using defined HSR_IP: ${HSR_IP}"
  elif [ -n "${ROBOT_NAME:-}" ]; then
    HSR_NAME="${ROBOT_NAME}"
    echo "Resolving global host name '${HSR_NAME}'..."
    HSR_IP="$(getent hosts "${HSR_NAME}" | awk '{ print $1 }')"
    if [ -z "${HSR_IP}" ]; then
      echo "Falling back to local host name '${HSR_NAME}.local'..."
      HSR_IP="$(avahi-resolve -4 --name "${HSR_NAME}.local" | awk '{ print $2 }')"
      if [ -z "${HSR_IP}" ]; then
        echo "Failed to resolve robot host name for ${HSR_NAME}."
        exit 1
      fi
    fi
    export HSR_IP
  else
    echo "[ERROR] Set one of HSR_IP or ROBOT_NAME for deploy mode."
    exit 1
  fi

  if [ -n "${ROS_IP:-}" ]; then
    echo "Using defined ROS_IP: ${ROS_IP}"
  else
    if command -v ifconfig >/dev/null 2>&1; then
      HOST_IP_LIST="$(ifconfig | grep 'inet ' | awk '{print $2}')"
    else
      HOST_IP_LIST="$(ip -4 -o addr show | awk '{print $4}' | cut -d/ -f1)"
    fi
    echo "Please choose ROS_IP from the following list:"
    select ip in ${HOST_IP_LIST}; do
      if [ -n "${ip}" ]; then
        export ROS_IP="${ip}"
        break
      fi
    done
  fi

  export ROS_MASTER_URI="http://${HSR_IP}:11311"
}

openpi_log_docker_context() {
  local caller="${1:-docker}"
  echo "${caller}: MODE=${MODE}"
  echo "${caller}: PROJECT=${PROJECT}"
  echo "${caller}: CONTAINER=${CONTAINER}"
  echo "${caller}: Using image tag: ${DEEP_IMAGE_TAG}"
  echo "${caller}: Using dockerfile: ${DEEP_DOCKERFILE}"
  echo "${caller}: OPENPI_PROJECT_PYTHON=${OPENPI_PROJECT_PYTHON}"
  echo "${caller}: OPENPI_ENABLE_ROS_INIT=${OPENPI_ENABLE_ROS_INIT}"
  echo "${caller}: DEEP_DATASET_PATH=${DEEP_DATASET_PATH}"
  echo "${caller}: CACHE_HOST_DIR=${DEEP_CACHE_ROOT_HOST}"
  if [ -n "${HSR_IP:-}" ]; then
    echo "${caller}: HSR_IP=${HSR_IP}"
  fi
  if [ -n "${ROS_IP:-}" ]; then
    echo "${caller}: ROS_IP=${ROS_IP}"
  fi
  if [ -n "${ROS_MASTER_URI:-}" ]; then
    echo "${caller}: ROS_MASTER_URI=${ROS_MASTER_URI}"
  fi
}
