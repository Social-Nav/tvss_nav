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
        rospy.loginfo("Costmap updater node launched...")

        # Parameters
        self.map_frame = rospy.get_param("~map_frame", "map")
        self.costmap_topic = rospy.get_param("~costmap_topic", "/rto/move_base_flex/global_costmap/costmap")
        self.output_topic = rospy.get_param("~output_topic", "/local_costmap_processed")
        self.cloud_prefix = rospy.get_param("~cloud_prefix", "/segmented_cloud/")

        self.min_height = rospy.get_param("~min_height", 0.1)
        self.max_height = rospy.get_param("~max_height", 2.0)
        self.min_distance = rospy.get_param("~min_distance", 0.1)
        self.max_distance = rospy.get_param("~max_distance", 10.0)
        self.cleanup_threshold = rospy.get_param("~cleanup_threshold", 2.0)

        # TF + sync
        self.tf_listener = tf.TransformListener()
        self.lock = threading.Lock()

        # ROS I/O
        self.costmap_sub = rospy.Subscriber(
            self.costmap_topic, OccupancyGrid, self.costmap_callback)
        self.costmap_pub = rospy.Publisher(
            self.output_topic, OccupancyGrid, queue_size=10)

        self.segmented_cloud_subs = {}
        self.last_cloud_time = {}
        self.segmented_pointclouds = {}

        self.raw_costmap = None
        self.map_info = None
        self.latest_costmap_time = 0

        # Periodic callbacks
        rospy.Timer(rospy.Duration(1.0), self.update_segmented_clouds)
        rospy.Timer(rospy.Duration(4.0), self.cleanup_old_segments)

    def update_segmented_clouds(self, event):
        all_topics = rospy.get_published_topics()
        active_clouds = {t[0] for t in all_topics if t[0].startswith(self.cloud_prefix)}

        for topic in active_clouds:
            instance_id = topic.split("/")[-1]
            if instance_id not in self.segmented_cloud_subs and instance_id != "0":
                rospy.loginfo(f"Subscribing to segmented cloud: {topic}")
                self.segmented_cloud_subs[instance_id] = rospy.Subscriber(
                    topic, PointCloud2, self.segmented_cloud_callback, callback_args=instance_id)

    def segmented_cloud_callback(self, msg, instance_id):
        cloud_frame = msg.header.frame_id
        timestamp = msg.header.stamp
        self.last_cloud_time[instance_id] = timestamp.to_sec()

        with self.lock:
            self.segmented_pointclouds[instance_id] = []

        try:
            self.tf_listener.waitForTransform(
                self.map_frame, cloud_frame, timestamp, rospy.Duration(1.0))
        except (tf.Exception, tf.LookupException, tf.ConnectivityException):
            rospy.logwarn(f"TF transform failed: {cloud_frame} -> {self.map_frame} at {timestamp.to_sec()}")
            return

        for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            x, y, z = p[:3]
            distance = np.sqrt(x**2 + y**2)
            if distance < self.min_distance or distance > self.max_distance:
                continue

            point = PointStamped()
            point.header.frame_id = cloud_frame
            point.header.stamp = timestamp
            point.point.x = x
            point.point.y = y
            point.point.z = z

            try:
                transformed_point = self.tf_listener.transformPoint(self.map_frame, point)
                with self.lock:
                    if instance_id in self.segmented_pointclouds:
                        self.segmented_pointclouds[instance_id].append(
                            (transformed_point.point.x, transformed_point.point.y, transformed_point.point.z)
                        )
            except (tf.Exception, tf.LookupException, tf.ConnectivityException):
                rospy.logwarn("TF transform failed while converting point.")
                continue

        self.update_costmap()

    def cleanup_old_segments(self, event):
        now = rospy.Time.now().to_sec()
        expired = [inst for inst, t in self.last_cloud_time.items() if now - t > self.cleanup_threshold]

        with self.lock:
            for instance_id in expired:
                rospy.logwarn(f"Removing stale segmented cloud: {instance_id}")
                del self.segmented_pointclouds[instance_id]
                del self.last_cloud_time[instance_id]
                self.segmented_cloud_subs[instance_id].unregister()
                del self.segmented_cloud_subs[instance_id]

        self.update_costmap()

    def costmap_callback(self, msg):
        self.raw_costmap = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        self.map_info = msg.info
        self.latest_costmap_time = msg.header.stamp
        self.update_costmap()

    def update_costmap(self):
        if self.raw_costmap is None or self.map_info is None:
            return

        updated_costmap = self.raw_costmap.copy()
        resolution = self.map_info.resolution
        origin_x = self.map_info.origin.position.x
        origin_y = self.map_info.origin.position.y

        with self.lock:
            for instance_id, points in list(self.segmented_pointclouds.items()):
                for x, y, z in points:
                    if z < self.min_height or z > self.max_height:
                        continue

                    grid_x = int((x - origin_x) / resolution)
                    grid_y = int((y - origin_y) / resolution)

                    if 0 <= grid_x < self.map_info.width and 0 <= grid_y < self.map_info.height:
                        updated_costmap[grid_y, grid_x] = 100

        updated_costmap = binary_dilation(updated_costmap, iterations=2).astype(np.int8) * 100

        new_costmap = OccupancyGrid()
        new_costmap.header.frame_id = self.map_frame
        new_costmap.header.stamp = self.latest_costmap_time
        new_costmap.info = self.map_info
        new_costmap.data = updated_costmap.flatten().tolist()

        self.costmap_pub.publish(new_costmap)


if __name__ == "__main__":
    node = CostmapUpdater()
    rospy.spin()
