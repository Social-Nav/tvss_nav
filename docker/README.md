## Quick Start

Build the container (GPU-enabled by default; from the `tvss_nav` package root). By default the image bootstraps the workspace via `install_lisn_ws.sh`. You can disable cloning/building during image build with `--build-arg RUN_BOOTSTRAP=0` or reuse existing checkouts with `--build-arg LISN_SKIP_FETCH=1`:

```bash
docker build --progress=plain -t lisn:latest -f Dockerfile.ros-torch .
# Skip bootstrap if you want a faster image build and plan to run install_lisn_ws.sh later:
# docker build --progress=plain -t lisn:latest -f Dockerfile.ros-torch --build-arg RUN_BOOTSTRAP=0 .
```

If you mount an existing host workspace into the container, keep the bootstrap fast and avoid recloning by combining `RUN_BOOTSTRAP=0` during build and running `install_lisn_ws.sh` inside the container with `LISN_SKIP_FETCH=1` (and optionally `LISN_SKIP_ROSDEP=1` if dependencies are already installed).

Run the container (automatically detects GPU availability). The helper script now attaches to the container by default; pass `--detach` to background it:

```bash
./docker/run.sh
# or: ./docker/run.sh --detach
# or run with custom command: ./docker/run.sh --cmd 'ls -la /root/lisn_ws'
```

Start a ROS master and visualize using Foxglove (example):

```bash
docker network create rosnet || true
docker run --rm --name roscore --hostname roscore --network rosnet -it -v ${PWD}/../../:/root/lisn_ws lisn:latest roscore
```

Run a publisher and the Foxglove bridge in separate containers (mapping Foxglove port):

```bash
docker run --rm -it --network rosnet --env 'ROS_MASTER_URI=http://roscore:11311/' -v ${PWD}/../../:/root/lisn_ws lisn:latest rostopic pub /chatter std_msgs/String 'data: hello' -r 1

docker run --rm -it --network rosnet --env 'ROS_MASTER_URI=http://roscore:11311/' -p 8765:8765 -v ${PWD}/../../:/root/lisn_ws lisn:latest roslaunch foxglove_bridge foxglove_bridge.launch
```

GPU-enabled `roscore` and Foxglove bridge using `lisn:latest`:

```bash
docker network create rosnet || true
docker run --gpus all --rm --name roscore --hostname roscore --network rosnet -it -v ${PWD}/../../:/root/lisn_ws lisn:latest roscore
docker run --gpus all --rm -it --network rosnet --env 'ROS_MASTER_URI=http://roscore:11311/' -p 8765:8765 -v ${PWD}/../../:/root/lisn_ws lisn:latest roslaunch foxglove_bridge foxglove_bridge.launch
```
```

Then in the host Foxglove app connect to `ws://localhost:8765`.

Start a ROS launch directly (example):

```bash
docker run --rm --net host --privileged \
  -v ${PWD}/../../:/root/lisn_ws \
  -p 8765:8765 \
  lisn:latest \
  roslaunch tvss_nav tvss_nav.launch
```

Notes:
- **Volume Mounting**: The container mounts the entire `lisn_ws` workspace directory, allowing live development and access to all files/results generated in the container.
- **Automatic GPU Detection**: The `run.sh` script automatically detects if GPUs are available and Docker-compatible, enabling GPU acceleration when possible.
- Use `--net host` for ROS communications between host and container (Linux).
- On macOS: run `docker run --net host` is not supported; instead use `--network rosnet` and `--env 'ROS_MASTER_URI=http://roscore:11311/'` or install Tailscale / other VPN overlay.
- If you need GPU support for PyTorch, either build from a CUDA base image or run the image with the NVIDIA runtime and install CUDA-aware PyTorch into the Conda env.
  - GPU image: build using `docker build --progress=plain -t lisn:latest -f Dockerfile.ros-torch .` and run with `--gpus all`.
  - For older Docker setups, you can use `--runtime=nvidia` instead of `--gpus all`.
- To use the Python ML features run `conda activate lisn` inside the container (the entrypoint activates the env when present).
- To visualize container topics in Foxglove: run the container with `-p 8765:8765` and start the `foxglove_bridge` node. Open the Foxglove app and connect to `ws://localhost:8765`.
- To use `rviz` or other GUI tools: mount the X11 socket and set `-e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix` and run the GUI app in the container. For macOS, use XQuartz and `socat` to proxy the X11 socket.
