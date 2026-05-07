#!/bin/bash
#PBS -q rt_HG
#PBS -P gch51606
#PBS -l select=1
#PBS -l walltime=01:30:00
#PBS -j oe

if [ -z "$WORKING_DIR" ]; then
  echo "[ERROR]: WORKING_DIR is not set."
  echo "[INFO]: Set WORKING_DIR like this:"
  echo "[INFO]: export WORKING_DIR=/path/to/AiroPi"
  exit 1
fi

module purge

module load singularitypro/4.1.7

module load cuda/12.4

# Define the path to the definition file and the output image
DEF_FILE="docker/openpi.def"
IMAGE_FILE="airopi.sif"

# Build the Singularity image
cd $WORKING_DIR
echo "Building Singularity image from $DEF_FILE..."
singularity build --fakeroot --nv $IMAGE_FILE $DEF_FILE
