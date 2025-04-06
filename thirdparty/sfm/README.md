# SFM Local Controller

由于arena的代码库启动文件耦合程度比较大，因此需要将一部分配置文件拷贝到arena的目录下,具体如下：

1. 将 `./copy/mbf_sfm.launch` 拷贝到 `~/arena_ws/src/arena/simulation-setup/launch/mbf/planners/local/mbf_sfm.launch`
2. 将 `./copy/sfm_local_planner_params.yaml` 拷贝到 `~/arena_ws/src/arena/simulation-setup/configs/mbf/local/sfm_local_planner_params.yaml`

## 编译

```
cd thirdparty/sfm/dep/lightsfm
make
sudo make install

catkin build
```

## 运行

```
cd tmux/arena_sfm
tmuxinator
```