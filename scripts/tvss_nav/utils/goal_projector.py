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
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import math
from tf.transformations import quaternion_from_euler
class GoalProjectorNode:
    def __init__(self):
        rospy.init_node('goal_projector')
        rospy.loginfo("Goal Projector Node initialized.")

        self.bridge = CvBridge()
        self.camera_info = None

        # === Load parameters ===
        camera_info_topic = rospy.get_param('/tvss_nav/color_info_topic', '/camera/color/camera_info')
        depth_image_topic = rospy.get_param('/tvss_nav/aligned_depth_topic', '/camera/aligned_depth_to_color/image_raw')
        pixel_goal_topic = rospy.get_param('/tvss_nav/pixel_goal_topic', '/pixel_subgoal')
        goal_pose_topic = rospy.get_param('/tvss_nav/goal_pose_topic', '/goalpose')
        target_frame = rospy.get_param('~target_frame', 'map')

        # === Initialize TF2 ===
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # === Subscribe to camera intrinsic info ===
        rospy.Subscriber(camera_info_topic, CameraInfo, self.camera_info_callback, queue_size=1)

        # === Synchronize depth image and pixel goal messages ===
        depth_sub = message_filters.Subscriber(depth_image_topic, Image)
        pixel_goal_sub = message_filters.Subscriber(pixel_goal_topic, PointStamped)

        ts = message_filters.TimeSynchronizer([depth_sub, pixel_goal_sub], queue_size=1000)
        # self.subgoal_pub = rospy.Publisher('/jackal/current_subgoal', PoseStamped, queue_size=1)
        ts.registerCallback(self.synced_callback)

        # === Publisher for 3D goal in target frame ===
        self.pub = rospy.Publisher(goal_pose_topic, PoseStamped, queue_size=1)
        self.target_frame = target_frame

        self.mb_ac = actionlib.SimpleActionClient('/jackal/move_base', MoveBaseAction)
        rospy.loginfo("Waiting for move_base action server…")
        self.mb_ac.wait_for_server()
        rospy.loginfo("Connected to move_base.")

    def camera_info_callback(self, msg):
        if self.camera_info is None:
            self.camera_info = msg
            rospy.loginfo("Camera intrinsic parameters received.")

    def synced_callback(self, depth_msg, pixel_goal_msg):
        if self.camera_info is None:
            rospy.logwarn("Waiting for camera info...")
            return

        try:
            depth_img = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

            u = int(pixel_goal_msg.point.x)
            v = int(pixel_goal_msg.point.y)

            if not (0 <= u < depth_img.shape[1] and 0 <= v < depth_img.shape[0]):
                rospy.logwarn("Pixel goal is out of image bounds.")
                return

            depth = depth_img[v, u]
            if depth <= 0:
                rospy.logwarn("Invalid depth value at pixel.")
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
            print("point_cam.point:",point_cam.point)
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.target_frame,
                    point_cam.header.frame_id,
                    point_cam.header.stamp,
                    timeout=rospy.Duration(1.0)
                )
                print("point_cam", point_cam)
                print("Transform found:", transform)
                point_map = tf2_geometry_msgs.do_transform_point(point_cam, transform)

                goal_msg = PoseStamped()
                goal_msg.header.frame_id = self.target_frame
                goal_msg.header.stamp = rospy.Time.now()
                goal_msg.pose.position = point_map.point
                goal_msg.pose.position.z = 0.0
                goal_msg.pose.orientation.w = 1.0
                dx = -goal_msg.pose.position.x
                dy = -goal_msg.pose.position.y
                yaw = math.atan2(dy, dx)
                q = quaternion_from_euler(0.0, 0.0, yaw)
                goal_msg.pose.orientation.x = q[0]
                goal_msg.pose.orientation.y = q[1]
                goal_msg.pose.orientation.z = q[2]
                goal_msg.pose.orientation.w = q[3]

                self.pub.publish(goal_msg)
            except Exception as e:
                rospy.logwarn(f"Failed to transform point to '{self.target_frame}': {e}")

        except Exception as e:
            rospy.logerr(f"Exception during callback execution: {e}")

if __name__ == '__main__':
    try:
        GoalProjectorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass