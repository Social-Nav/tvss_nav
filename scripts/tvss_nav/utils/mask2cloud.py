#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, CameraInfo, RegionOfInterest, PointCloud2, PointField
from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge

bridge = CvBridge()

depth_image = None
cloud_pubs = {}

def get_camera_info(stamp, frame_id):
    """
    Returns a CameraInfo message with predefined intrinsic parameters.
    """
    ci = CameraInfo()
    ci.header = Header()
    ci.header.stamp = stamp
    ci.header.frame_id = frame_id
    ci.height = 480
    ci.width = 640
    ci.distortion_model = "plumb_bob"
    ci.D = []
    ci.K = [415.6921691894531, 0.0, 320.0, 0.0, 415.6921691894531, 240.0, 0.0, 0.0, 1.0]
    ci.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    ci.P = [415.6921691894531, 0.0, 320.0, 0.0, 0.0, 415.6921691894531, 240.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    ci.binning_x = 0
    ci.binning_y = 0
    ci.roi = RegionOfInterest()
    ci.roi.x_offset = 0
    ci.roi.y_offset = 0
    ci.roi.height = 0
    ci.roi.width = 0
    ci.roi.do_rectify = False
    return ci

def erode_depth(depth_image, kernel_size=3, iterations=1):
    """ 
    Applies erosion to the depth image to reduce edge noise.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    depth_mask = np.isfinite(depth_image).astype(np.uint8)  # Only erode finite values (ignore infinite depth)
    eroded_mask = cv2.erode(depth_mask, kernel, iterations=iterations)

    depth_image_filtered = depth_image * eroded_mask
    return depth_image_filtered

def depth_callback(msg):
    """Receives depth image and stores it globally."""
    global depth_image
    depth_image = bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")

def depth_to_pointcloud(instance_id, depth_image, camera_info):
    """ Converts a depth image into a 3D point cloud and publishes it. """
    camera_matrix = np.array(camera_info.K).reshape(3, 3)

    height, width = depth_image.shape

    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]  # Focal length
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]  # Principal point

    u, v = np.meshgrid(np.arange(width), np.arange(height))

    Z = depth_image
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    valid = (Z > 0) & (Z < np.inf)
    points = np.stack((X[valid], Y[valid], Z[valid]), axis=-1)

    header = rospy.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = "center_depth_optical_frame"

    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]

    cloud_msg = pc2.create_cloud(header, fields, points)
    
    publish_pointcloud(instance_id, cloud_msg)

def publish_pointcloud(instance_id, cloud_msg):
    """ Publishes the point cloud under a specific instance topic. """
    topic_name = f"/instance_cloud/{instance_id}"
    if instance_id not in cloud_pubs:
        cloud_pubs[instance_id] = rospy.Publisher(topic_name, PointCloud2, queue_size=10)
        rospy.loginfo(f"Created new topic: {topic_name}")
    cloud_pubs[instance_id].publish(cloud_msg)

def mask_callback(msg):
    """Receives an instance mask and processes the corresponding depth image."""
    frame_id = "center_depth_optical_frame"
    ci = get_camera_info(msg.header.stamp, frame_id)
    kernel_size = 3

    global depth_image
    try:
        np_arr = np.frombuffer(msg.data, dtype=np.uint8)
        mask = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
        all_mask = mask.copy()

        unique_ids = np.unique(mask)
        unique_ids = unique_ids[unique_ids > 0]
        if mask is None:
            rospy.logerr("Failed to decode mask!")
            return
        
        if depth_image is None:
            rospy.logwarn("Waiting for depth image data...")
            return

        max_depth = np.inf
        
        for instance_id in unique_ids:
            masked_depth = np.where(mask == instance_id, depth_image, 0)
            masked_depth = erode_depth(masked_depth, kernel_size, iterations=1)
            depth_to_pointcloud(instance_id, masked_depth, ci)

        all_masked_depth = np.where(all_mask > 0, depth_image, max_depth)  # Keep original depth inside mask, set background to max depth
        all_masked_depth = erode_depth(all_masked_depth, kernel_size, iterations=1)  # Apply erosion
        depth_to_pointcloud(0, all_masked_depth, ci)
        
        masked_depth_msg = bridge.cv2_to_imgmsg(all_masked_depth, encoding="32FC1")
        masked_depth_msg.header.frame_id = frame_id
        masked_depth_pub.publish(masked_depth_msg)
        
        info_pub.publish(ci)

    except Exception as e:
        rospy.logerr(f"Processing failed: {e}")

if __name__ == "__main__":
    rospy.init_node("mask2cloud_processor")

    rospy.Subscriber("/segmented_image/mask", Image, mask_callback)
    rospy.Subscriber("/center_depth_sync/image_raw", Image, depth_callback)

    info_pub = rospy.Publisher('/masked_depth_image/camera_info', CameraInfo, queue_size=10)
    masked_depth_pub = rospy.Publisher("/masked_depth_image/image_raw", Image, queue_size=10)

    rospy.spin()
