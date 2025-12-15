#!/usr/bin/env bash
set -euo pipefail

# Helper script to run the GPU-enabled lisn container with recommended flags.
# Usage: ./docker/run.sh [--image tag] [--cmd 'bash']

IMAGE=${IMAGE:-lisn:latest}
CMD="bash"
DETACH=0

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --image)
      IMAGE="$2"
      shift 2
      ;;
    --cmd)
      CMD="$2"
      shift 2
      ;;
    --detach)
      DETACH=1
      shift 1
      ;;
    --no-detach)
      DETACH=0
      shift 1
      ;;
    *)
      # Unknown option, assume it's the command
      CMD="$*"
      break
      ;;
  esac
done

WORKDIR=${LISN_WS_DIR:-/root/lisn_ws}

# Find the workspace root (parent of src directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_DIR="$(dirname "$SCRIPT_DIR")"  # /home/ubuntu/lisn_ws/src/tvss_nav
SRC_DIR="$(dirname "$DOCKER_DIR")"     # /home/ubuntu/lisn_ws/src
LISN_WS_DIR="$(dirname "$SRC_DIR")"    # /home/ubuntu/lisn_ws

# Check if NVIDIA GPU is available and Docker can use it
GPU_ARGS=""
if command -v nvidia-smi >/dev/null 2>&1; then
  if nvidia-smi --query-gpu=name --format=csv,noheader,nounits >/dev/null 2>&1; then
    # Test if Docker can actually use GPUs
    if docker run --rm --gpus all ubuntu:20.04 echo "GPU test successful" >/dev/null 2>&1; then
      GPU_ARGS="--gpus all"
      echo "GPU support detected and Docker-compatible, enabling GPU acceleration"
    else
      echo "GPU detected but Docker cannot access it, running without GPU support"
    fi
  else
    echo "NVIDIA driver detected but GPU not accessible, running without GPU support"
  fi
else
  echo "No NVIDIA GPU drivers detected, running without GPU support"
fi

RUN_FLAGS="-it --rm"
if [[ ${DETACH} -eq 1 ]]; then
  RUN_FLAGS="--rm -d"
fi

docker run ${GPU_ARGS} ${RUN_FLAGS} --net host --privileged \
  -v "${LISN_WS_DIR}:${WORKDIR}" \
  ${DISPLAY:+-e DISPLAY="$DISPLAY"} \
  ${DISPLAY:+-v /tmp/.X11-unix:/tmp/.X11-unix} \
  -p 8765:8765 \
  ${IMAGE} ${CMD}
