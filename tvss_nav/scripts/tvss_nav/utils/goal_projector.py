#!/usr/bin/env python
import rospy
import message_filters
import tf2_ros
import tf2_geometry_msgs
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped, PoseStamped
import cv2
import numpy as np
from cv_bridge import CvBridge

class GoalProjectorNode:
    def __init__(self):
        rospy.init_node('goal_projector')
        rospy.loginfo("Goal Projector Node started.")

        self.bridge = CvBridge()
        self.camera_info = None

        # === 参数读取 ===
        camera_info_topic = rospy.get_param('~camera_info_topic', '/camera/color/camera_info')
        depth_image_topic = rospy.get_param('~depth_image_topic', '/camera/aligned_depth_to_color/image_raw')
        pixel_goal_topic = rospy.get_param('~pixel_goal_topic', '/pixel_subgoal')
        goal_pose_topic = rospy.get_param('~goal_pose_topic', '/goalpose')
        target_frame = rospy.get_param('~target_frame', 'map')

        # === TF2 ===
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # === 订阅相机内参 ===
        rospy.Subscriber(camera_info_topic, CameraInfo, self.camera_info_callback, queue_size=1)

        # === 同步深度图和目标像素点 ===
        depth_sub = message_filters.Subscriber(depth_image_topic, Image)
        pixel_goal_sub = message_filters.Subscriber(pixel_goal_topic, PointStamped)

        ts = message_filters.TimeSynchronizer([depth_sub, pixel_goal_sub], queue_size=1000)
        ts.registerCallback(self.synced_callback)

        # === 发布目标位姿 ===
        self.pub = rospy.Publisher(goal_pose_topic, PoseStamped, queue_size=1)
        self.target_frame = target_frame

    def camera_info_callback(self, msg):
        if self.camera_info is None:
            self.camera_info = msg
            rospy.loginfo("Camera info received.")

    def synced_callback(self, depth_msg, pixel_goal_msg):
        if self.camera_info is None:
            rospy.logwarn("Camera info not yet received.")
            return

        try:
            depth_img = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

            u = int(pixel_goal_msg.point.x)
            v = int(pixel_goal_msg.point.y)

            if not (0 <= u < depth_img.shape[1] and 0 <= v < depth_img.shape[0]):
                rospy.logwarn("Pixel goal out of image bounds.")
                return

            depth = depth_img[v, u]
            if depth <= 0:
                rospy.logwarn("Invalid depth at pixel.")
                return

            K = np.array(self.camera_info.K).reshape(3, 3)
            D = np.array(self.camera_info.D)
            uv = np.array([[[u, v]]], dtype=np.float32)
            undistorted = cv2.undistortPoints(uv, K, D)
            x_n, y_n = undistorted[0][0]

            x, y, z = x_n * depth, y_n * depth, depth

            point_cam = PointStamped()
            point_cam.header = pixel_goal_msg.header
            point_cam.point.x = x
            point_cam.point.y = y
            point_cam.point.z = z

            try:
                transform = self.tf_buffer.lookup_transform(
                    self.target_frame,
                    point_cam.header.frame_id,
                    point_cam.header.stamp,
                    timeout=rospy.Duration(1.0)
                )
                point_map = tf2_geometry_msgs.do_transform_point(point_cam, transform)

                goal_msg = PoseStamped()
                goal_msg.header.frame_id = self.target_frame
                goal_msg.header.stamp = rospy.Time.now()
                goal_msg.pose.position = point_map.point
                goal_msg.pose.position.z = 0.0
                goal_msg.pose.orientation.w = 1.0

                self.pub.publish(goal_msg)
                rospy.loginfo(f"Published 3D goal in {self.target_frame}: ({point_map.point.x:.2f}, {point_map.point.y:.2f}, {point_map.point.z:.2f})")

            except Exception as e:
                rospy.logwarn(f"TF transform to {self.target_frame} failed: {e}")

        except Exception as e:
            rospy.logerr(f"Error processing callback: {e}")

if __name__ == '__main__':
    try:
        GoalProjectorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
