#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import tf
import threading
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
from scipy.ndimage import binary_dilation
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import PointStamped


class CostmapUpdater:
    def __init__(self):
        rospy.init_node("costmap_updater", anonymous=True)
        self.tf_listener = tf.TransformListener()
        self.lock = threading.Lock()  # Thread lock to prevent concurrent dictionary modifications

        # Subscribe to the original costmap
        self.costmap_sub = rospy.Subscriber('/move_base/local_costmap/costmap', OccupancyGrid, self.costmap_callback)

        # Costmap publisher
        self.costmap_pub = rospy.Publisher('/local_costmap_processed', OccupancyGrid, queue_size=10)

        # Maintain dynamically subscribed `instance_cloud`
        self.instance_cloud_subs = {}  
        self.last_update_time = {}  # Record the latest `timestamp` for each `instance_id`
        self.cleanup_threshold = 2.0  # Remove `instance` if not updated for more than N seconds

        # Costmap data
        self.raw_costmap = None
        self.map_info = None
        self.instance_pointclouds = {}  # Store point cloud data for each `instance_cloud`

        # Constraint parameters
        self.min_height = 0.1
        self.max_height = 2.0
        self.min_distance = 0.1
        self.max_distance = 10.0

        # Periodically check for `/instance_cloud/{id}` topics
        rospy.Timer(rospy.Duration(1.0), self.update_instance_clouds)
        rospy.Timer(rospy.Duration(2.0), self.cleanup_old_instances)

    def update_instance_clouds(self, event):
        """Dynamically check for `/instance_cloud/{id}` topics"""
        all_topics = rospy.get_published_topics()
        active_instance_clouds = {t[0] for t in all_topics if t[0].startswith("/instance_cloud/")}

        # Subscribe to new `instance_cloud`
        for topic in active_instance_clouds:
            instance_id = topic.split("/")[-1]
            if instance_id not in self.instance_cloud_subs and instance_id != "0":
                rospy.loginfo(f"Subscribing to new instance_cloud: {topic}")
                self.instance_cloud_subs[instance_id] = rospy.Subscriber(
                    topic, PointCloud2, self.instance_cloud_callback, callback_args=instance_id
                )

    def instance_cloud_callback(self, msg, instance_id):
        """Process `instance_cloud` point clouds and perform TF coordinate transformation"""
        cloud_frame = msg.header.frame_id
        self.last_update_time[instance_id] = msg.header.stamp.to_sec()  # Record the latest `timestamp`

        with self.lock:  # Lock to prevent dictionary modifications
            self.instance_pointclouds[instance_id] = []  # Clear old data

        try:
            self.tf_listener.waitForTransform("map", cloud_frame, rospy.Time(0), rospy.Duration(1.0))
        except (tf.Exception, tf.LookupException, tf.ConnectivityException):
            rospy.logwarn(f"TF transformation failed: {cloud_frame} -> map")
            return

        for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            x, y, z = p[:3]
            distance = np.sqrt(x**2 + y**2)
            if distance < self.min_distance or distance > self.max_distance:
                continue

            point = PointStamped()
            point.header.frame_id = cloud_frame
            point.header.stamp = rospy.Time(0)
            point.point.x = x
            point.point.y = y
            point.point.z = z

            try:
                transformed_point = self.tf_listener.transformPoint("map", point)
                with self.lock:  # 🔒 Lock to prevent modifications by `cleanup_old_instances`
                    if instance_id in self.instance_pointclouds:
                        self.instance_pointclouds[instance_id].append(
                            (transformed_point.point.x, transformed_point.point.y, transformed_point.point.z)
                        )
            except (tf.Exception, tf.LookupException, tf.ConnectivityException):
                rospy.logwarn("TF coordinate transformation failed")
                continue

        self.update_costmap()

    def cleanup_old_instances(self, event):
        """Remove `instance_cloud` that have not been updated for a long time"""
        now = rospy.Time.now().to_sec()
        expired_instances = [inst for inst, t in self.last_update_time.items() if now - t > self.cleanup_threshold]

        with self.lock:  # 🔒 Lock to prevent `update_costmap()` from modifying the dictionary concurrently
            for instance_id in expired_instances:
                rospy.logwarn(f"Removing expired instance_cloud: {instance_id}")
                del self.instance_pointclouds[instance_id]
                del self.last_update_time[instance_id]
                self.instance_cloud_subs[instance_id].unregister()
                del self.instance_cloud_subs[instance_id]

        self.update_costmap()

    def costmap_callback(self, msg):
        """Process the original costmap"""
        self.raw_costmap = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        self.map_info = msg.info
        self.update_costmap()

    def update_costmap(self):
        """Update the costmap by adding obstacles from point clouds"""
        if self.raw_costmap is None or self.map_info is None:
            return

        updated_costmap = self.raw_costmap.copy()
        resolution = self.map_info.resolution
        origin_x = self.map_info.origin.position.x
        origin_y = self.map_info.origin.position.y

        with self.lock:  # 🔒 Lock to prevent `cleanup_old_instances()` from modifying `self.instance_pointclouds`
            for instance_id, pointcloud in list(self.instance_pointclouds.items()):  # ✅ Iterate over a copy
                for x, y, z in pointcloud:
                    if z < self.min_height or z > self.max_height:
                        continue  # Filter out points outside the height range

                    grid_x = int((x - origin_x) / resolution)
                    grid_y = int((y - origin_y) / resolution)

                    if 0 <= grid_x < self.map_info.width and 0 <= grid_y < self.map_info.height:
                        updated_costmap[grid_y, grid_x] = 100  # Mark as high-cost (obstacle)

        updated_costmap = binary_dilation(updated_costmap, iterations=2).astype(np.int8) * 100

        new_costmap = OccupancyGrid()
        new_costmap.header.frame_id = "map"
        new_costmap.header.stamp = rospy.Time.now()
        new_costmap.info = self.map_info
        new_costmap.data = updated_costmap.flatten().tolist()

        self.costmap_pub.publish(new_costmap)


if __name__ == "__main__":
    node = CostmapUpdater()
    rospy.spin()
