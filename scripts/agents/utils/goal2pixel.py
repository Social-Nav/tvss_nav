#!/usr/bin/env python
import rospy
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseArray
from std_msgs.msg import Int32MultiArray

class DummyGoal2Pixel:
    def __init__(self):
        rospy.init_node('goal2pixel', anonymous=False)
        
        # Subscribers
        rospy.Subscriber('/local_cost_map_raw', OccupancyGrid, self.local_cost_map_callback)
        rospy.Subscriber('/goal_samples', PoseArray, self.goal_samples_callback)
        
        # Publisher
        self.pixel_coordinate_pub = rospy.Publisher('/samples_pixel_coordinate', Int32MultiArray, queue_size=10)
        
        # Placeholder for latest local cost map
        self.local_cost_map = None
        
    def local_cost_map_callback(self, data):
        self.local_cost_map = data  # Store latest cost map data

    def goal_samples_callback(self, pose_array):
        if not self.local_cost_map:
            rospy.logwarn("No local cost map data received yet.")
            return
        
        for pose in pose_array.poses:
            # return the pixel coordinate of the goal on the local cost map
            pixel_x = int(pose.position.x / self.local_cost_map.info.resolution)
            pixel_y = int(pose.position.y / self.local_cost_map.info.resolution)
            pixel_coords = Int32MultiArray(data=[pixel_x, pixel_y])
            self.pixel_coordinate_pub.publish(pixel_coords)

if __name__ == '__main__':
    DummyGoal2Pixel()
    if not rospy.is_shutdown():
        rospy.spin()
