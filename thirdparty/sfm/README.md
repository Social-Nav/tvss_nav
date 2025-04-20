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

## 注意

修改了lightsfm头文件的参数后，要想让参数在sfm_local_controller plugin中生效，需要清除sfm_local_controller plugin的缓存，具体如下:
```
cd thirdparty/sfm/dep/lightsfm
...do some changes...
make
sudo make install
catkin clean sfm_local_controller
catkin build sfm_local_controller
```