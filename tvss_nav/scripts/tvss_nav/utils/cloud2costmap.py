#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
from nav_msgs.msg import OccupancyGrid
from tvss_nav.msg import SemanticInstanceArray
from sensor_msgs.msg import PointCloud2
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
import tf2_ros
import tf2_py
from scipy.ndimage import gaussian_filter


class InstanceCostmapUpdater:
    def __init__(self):
        rospy.init_node("instance_costmap_updater", anonymous=False)
        rospy.loginfo("Instance Costmap Updater node launched...")

        # Parameters
        self.map_frame = rospy.get_param("~map_frame", "map")
        self.output_topic = rospy.get_param("~output_topic", "/local_costmap_processed")
        self.min_height = rospy.get_param("~min_height", 0.1)
        self.max_height = rospy.get_param("~max_height", 10.0)

        while not rospy.has_param("model") and not rospy.is_shutdown():
            rospy.loginfo("Waiting for model parameter...")
            rospy.sleep(0.1)

        self.prefix = rospy.get_param("model", "").strip("/")
        self.origin_costmap_topic = rospy.get_param(
            "~costmap_topic",
            f"/{self.prefix}/move_base_flex/global_costmap/costmap" if self.prefix else "/move_base_flex/global_costmap/costmap"
        )

        self.map_width = rospy.get_param("~map_width", 200)
        self.map_height = rospy.get_param("~map_height", 200)
        self.map_resolution = rospy.get_param("~map_resolution", 0.1)
        self.map_origin_x = rospy.get_param("~map_origin_x", -10.0)
        self.map_origin_y = rospy.get_param("~map_origin_y", -10.0)

        self.latest_map = None
        self.cached_global = None

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # ROS I/O
        self.instance_sub = rospy.Subscriber("/instance_array", SemanticInstanceArray, self.instance_callback)
        self.costmap_pub = rospy.Publisher(self.output_topic, OccupancyGrid, queue_size=1)
        self.global_sub = rospy.Subscriber(self.origin_costmap_topic, OccupancyGrid, self.global_callback)
        rospy.Timer(rospy.Duration(1.0), self.timer_update_global)

    def global_callback(self, msg):
        self.cached_global = msg

    def timer_update_global(self, event):
        if self.cached_global is not None:
            self.latest_map = self.cached_global

    def instance_callback(self, msg):
        if self.latest_map is None:
            rospy.logwarn_throttle(5.0, "Waiting for global map...")
            return

        base_grid = np.array(self.latest_map.data, dtype=np.int8).reshape(
            self.latest_map.info.height, self.latest_map.info.width
        )
        grid = np.copy(base_grid).astype(np.float32)

        for instance in msg.instances:
            cloud = instance.cloud
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.map_frame,
                    cloud.header.frame_id,
                    cloud.header.stamp,
                    rospy.Duration(1.0)
                )
                cloud = do_transform_cloud(cloud, transform)
            except (tf2_ros.LookupException, tf2_ros.ExtrapolationException):
                rospy.logwarn("TF transform failed, skipping instance.")
                continue

            resolution = self.latest_map.info.resolution
            origin_x = self.latest_map.info.origin.position.x
            origin_y = self.latest_map.info.origin.position.y
            width = self.latest_map.info.width
            height = self.latest_map.info.height

            mask = np.zeros((height, width), dtype=np.float32)

            for i in range(0, len(cloud.data), cloud.point_step):
                x = np.frombuffer(cloud.data[i + cloud.fields[0].offset:i + cloud.fields[0].offset + 4], dtype=np.float32)[0]
                y = np.frombuffer(cloud.data[i + cloud.fields[1].offset:i + cloud.fields[1].offset + 4], dtype=np.float32)[0]
                z = np.frombuffer(cloud.data[i + cloud.fields[2].offset:i + cloud.fields[2].offset + 4], dtype=np.float32)[0]

                if not (self.min_height <= z <= self.max_height):
                    continue

                grid_x = int((x - origin_x) / resolution)
                grid_y = int((y - origin_y) / resolution)

                if 0 <= grid_x < width and 0 <= grid_y < height:
                    mask[grid_y, grid_x] = 1.0

            if instance.inflation_radius > 0:
                sigma = instance.inflation_radius / resolution
                mask = gaussian_filter(mask, sigma=sigma)
                if instance.decay_rate > 0:
                    mask = np.clip(mask * instance.cost_value * instance.decay_rate, 0, 254)
                else:
                    mask *= instance.cost_value

                grid = np.maximum(grid, mask)

        costmap = OccupancyGrid()
        costmap.header.frame_id = self.map_frame
        costmap.header.stamp = msg.header.stamp
        costmap.info = self.latest_map.info
        costmap.data = grid.astype(np.int8).flatten().tolist()

        self.costmap_pub.publish(costmap)


if __name__ == "__main__":
    node = InstanceCostmapUpdater()
    rospy.spin()
