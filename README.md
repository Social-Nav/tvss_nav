# Task-Oriented Visual-Semantic Social Navigation

## 2025-03-05 Update

### Steps to run grounded sam 2 segmentation
1. Run the Unity World

2. Run the teleoporation node and rosbridge (rosbridge is for topic communication between ros and python script using conda env)
    ```bash
    cd ./third_party/tmux/sean_teleop
    tmuxinator
    ```

3. Run grounded sam2 in ros

    ```bash
    conda create -n gsam2 python=3.11
    conda activate gsam2
    # Follow the instructions on https://github.com/IDEA-Research/Grounded-SAM-2/blob/main/INSTALL.md to install the dependencies
    cd ./scripts/tvss_nav/tools/grounded_sam2
    python gsam2_ros.py
    ```

### Steps to run instance 3D mask generation and cost-map projection

1. run the instance mask->instance cloud processor ros node: 

    ```bash
    cd ./scripts/tvss_nav/utils
    python mask2cloud.py
    ```
2. run the cloud->costmap ros node: 

    ```bash
    cd ./scripts/tvss_nav/utils
    python cloud2costmap.py
    ```

## Nodes

1. subgoal sampler
    1. subscribe
        1. /local_cost_map_raw (type: nav_msgs::OccupancyGrid, note: obtain from move base, needs remapping)
    2. publish
        1. /goal_samples (type: geometry_msgs::PoseArray, frame: odom)
2. vlm
   
    Use `rosparam set` to change the behavior of local planner
    
    1. subscribe
        1. /task_description (str)
        2. /current_observation (type: sensor_msgs/Image or CompressedImage)
        3. /samples_pixel_coordinate
    2. publish
        1. /selected_visual_prompt
        2. /function_call
3. tools

Only Grounding SAM2 for now

1. subscribe
    1. /current_observation (type: sensor_msgs/Image or CompressedImage)
    2. /function_call
2. publish
    1. /image_masked_high_cost (Type: Image)
    2. /image_masked_middle_cost (Type: Image)
    3. /image_masked_low_cost (Type: Image)
    4. **/segmented_image/mask (Type: Image)**
    5. ….. Depends on the predefined levels
1. utils:

Topics depend on the actual realization of these methods

1. covert goal on costmap to pixel coordinate
    1. subscribe
        1. /local_cost_map_raw
        2. /goal_samples
    2. publish
        1. /samples_pixel_coordinate (int[2])
2. convert pixel cooridnate to goal on costmap
    1. subscribe
        1. /local_cost_map_raw
        2. /selected_visual_prompt
    2. publish
        1. /goal
3. convert image mask to costmap given predefined cost values on different classes
    1. subscribe
        1. /local_cost_map_raw
        2. /image_masked_low_cost
        3. ….
    2. publish
        1. /local_cost_map_processed
4. **convert instance image mask to 3D instance point cloud**
    1. **subscribe**
        1. **/segmented_image/mask**
        2. **/center_depth_sync/image_raw**
    2. **publish**
        1. **/masked_depth_image/camera_info**
        2. **/masked_depth_image/image_raw**
        3. **all topics under /instance_cloud group (e.g. /instance_cloud/1)**
5. **convert instance point cloud to costmap**
    1. **subscribe**
        1. **/move_base/local_costmap/costmap**
        2. **all topics under /instance_cloud group (e.g. /instance_cloud/1)**
    2. **publish**
        1. **/local_costmap_processed**