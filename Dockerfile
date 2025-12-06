ARG CUDA_TAG=11.8.0-cudnn8-runtime-ubuntu20.04
FROM nvidia/cuda:${CUDA_TAG}
LABEL com.nvidia.volumes.needed="nvidia_driver"
LABEL ros.distro="noetic"

ARG ROS_DISTRO=noetic
ARG LISN_WS_DIR=/root/lisn_ws
ARG CONDA_DIR=/opt/conda
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    ROS_DISTRO=${ROS_DISTRO} \
    LISN_WS_DIR=${LISN_WS_DIR} \
    CONDA_DIR=${CONDA_DIR} \
    PATH=${CONDA_DIR}/bin:${PATH}
ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,video,utility

SHELL ["/bin/bash", "-lc"]

# Install OS-level dependencies and ROS packages (Noetic installed via apt)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gnupg2 curl ca-certificates wget git sudo build-essential lsb-release \
        python3-pip python3-rosdep python3-catkin-tools python3-rosinstall python3-rosinstall-generator python3-wstool \
        cmake pkg-config libeigen3-dev libboost-all-dev libopencv-dev libpcl-dev locales \
        x11-apps mesa-utils \
    && rm -rf /var/lib/apt/lists/*

# ROS apt repository & packages
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | apt-key add - \
 && echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends ros-${ROS_DISTRO}-desktop-full ros-${ROS_DISTRO}-foxglove-bridge \
 && rm -rf /var/lib/apt/lists/*

# The base image already provides ROS Noetic (osrf/ros:noetic-desktop-full)
# Install additional ROS tools if needed via apt in a separate step.

# Initialize rosdep
RUN rosdep init || true && rosdep update || true

# Create workspace
RUN mkdir -p ${LISN_WS_DIR}/src
WORKDIR ${LISN_WS_DIR}

# Copy the local tvss_nav repository into the workspace
COPY . ${LISN_WS_DIR}/src/tvss_nav

# Default to a small number of parallel jobs for deterministic builds
ENV MAKEFLAGS="-j2"

# Bootstrap and build the workspace using the included installer
# The installer clones dependent repos and builds lightsfm which may require additional libraries.
RUN /bin/bash -lc \
    set -euo pipefail; \
    source /opt/ros/${ROS_DISTRO}/setup.bash; \
    cd ${LISN_WS_DIR}; \
    bash src/tvss_nav/install_lisn_ws.sh || (echo "Workspace install may have failed. Check build output"; exit 1)

# Build the workspace (catkin tools)
RUN /bin/bash -lc \
    source /opt/ros/${ROS_DISTRO}/setup.bash; \
    cd ${LISN_WS_DIR}; \
    catkin config --source-space src || true; \
    catkin build || true

# Install Miniconda for Python 3.11 environment (recommended by README)
ENV MINICONDA_INSTALLER=Miniconda3-latest-Linux-x86_64.sh
RUN wget --quiet https://repo.anaconda.com/miniconda/${MINICONDA_INSTALLER} -O /tmp/${MINICONDA_INSTALLER} \
    && bash /tmp/${MINICONDA_INSTALLER} -b -p ${CONDA_DIR} \
    && rm /tmp/${MINICONDA_INSTALLER} \
    && ${CONDA_DIR}/bin/conda init bash || true

# Create the 'lisn' conda environment with Python 3.11 and install Python deps
RUN ${CONDA_DIR}/bin/conda create -y -n lisn python=3.11 \
    && ${CONDA_DIR}/bin/conda install -y -n lisn -c conda-forge mamba \
    && ${CONDA_DIR}/bin/conda run -n lisn mamba install -y -c pytorch -c nvidia -c conda-forge pytorch torchvision torchaudio pytorch-cuda=11.8 \
    && ${CONDA_DIR}/bin/conda run -n lisn pip install --upgrade pip setuptools wheel \
    && ${CONDA_DIR}/bin/conda run -n lisn pip install -r ${LISN_WS_DIR}/src/tvss_nav/requirements.txt || true

# Change to root workspace and source the workspaces by default
ENV ROS_ENTRYPOINT=/usr/local/bin/ros_entrypoint.sh
COPY docker/entrypoint.sh ${ROS_ENTRYPOINT}
RUN chmod +x ${ROS_ENTRYPOINT}

EXPOSE 11311 8765
VOLUME ["/root/lisn_ws"]

ENTRYPOINT ["/usr/local/bin/ros_entrypoint.sh"]
CMD ["bash"]

# Default environment variables useful for GPU-enabled containers
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,video,utility
