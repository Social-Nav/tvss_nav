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

        # TF2 buffer and listener for transform to map
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Subscribe to camera intrinsic parameters (only once)
        rospy.Subscriber('/camera/color/camera_info', CameraInfo, self.camera_info_callback, queue_size=1)

        # Synchronize depth image and pixel goal
        depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image)
        pixel_goal_sub = message_filters.Subscriber('/pixel_subgoal', PointStamped)

        ts = message_filters.TimeSynchronizer(
            [depth_sub, pixel_goal_sub],
            queue_size=1000
        )
        ts.registerCallback(self.synced_callback)

        # Publish 3D goal in map frame as a PoseStamped message
        self.pub = rospy.Publisher('/goalpose', PoseStamped, queue_size=1)

    def camera_info_callback(self, msg):
        if self.camera_info is None:
            self.camera_info = msg
            rospy.loginfo("Camera info received.")

    def synced_callback(self, depth_msg, pixel_goal_msg):
        if self.camera_info is None:
            rospy.logwarn("Camera info not yet received.")
            return

        try:
            # Convert depth image
            depth_img = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

            u = int(pixel_goal_msg.point.x)
            v = int(pixel_goal_msg.point.y)

            if not (0 <= u < depth_img.shape[1] and 0 <= v < depth_img.shape[0]):
                rospy.logwarn("Pixel goal out of image bounds.")
                return

            depth = depth_img[v, u] / 1.0  # Keep as meters (adapt if depth is in mm)
            if depth <= 0:
                rospy.logwarn("Invalid depth at pixel.")
                return

            # Camera intrinsics and distortion coefficients
            K = np.array(self.camera_info.K).reshape(3, 3)
            D = np.array(self.camera_info.D)

            # Undistort and normalize pixel coordinates
            uv = np.array([[[u, v]]], dtype=np.float32)
            undistorted = cv2.undistortPoints(uv, K, D)
            x_n = undistorted[0][0][0]
            y_n = undistorted[0][0][1]

            x = x_n * depth
            y = y_n * depth
            z = depth

            # Create PointStamped in camera frame
            point_cam = PointStamped()
            point_cam.header = pixel_goal_msg.header  # Should be in camera frame
            point_cam.point.x = x
            point_cam.point.y = y
            point_cam.point.z = z

            try:
                # Transform point to map frame
                transform = self.tf_buffer.lookup_transform(
                    target_frame="map",
                    source_frame=point_cam.header.frame_id,
                    time=point_cam.header.stamp,
                    timeout=rospy.Duration(1.0)
                )

                point_map = tf2_geometry_msgs.do_transform_point(point_cam, transform)

                # Create PoseStamped in map frame
                goal_msg = PoseStamped()
                goal_msg.header.frame_id = "map"
                goal_msg.header.stamp = rospy.Time.now()
                goal_msg.pose.position = point_map.point
                goal_msg.pose.position.z = 0.0  # Set z to 0 for 2D navigation
                goal_msg.pose.orientation.w = 1.0  # Identity quaternion
                goal_msg.pose.orientation.x = 0.0
                goal_msg.pose.orientation.y = 0.0
                goal_msg.pose.orientation.z = 0.0

                self.pub.publish(goal_msg)
                rospy.loginfo(f"Published 3D goal in map frame: ({point_map.point.x:.2f}, {point_map.point.y:.2f}, {point_map.point.z:.2f})")

            except Exception as e:
                rospy.logwarn(f"TF transform to map failed: {e}")

        except Exception as e:
            rospy.logerr(f"Error processing callback: {e}")

if __name__ == '__main__':
    try:
        GoalProjectorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
