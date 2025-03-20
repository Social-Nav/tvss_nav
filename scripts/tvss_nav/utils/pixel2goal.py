#!/usr/bin/env python
import rospy
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Int32MultiArray, String

class DummyPixel2Goal:
    def __init__(self):
        rospy.init_node('pixel2goal', anonymous=True)
        
        # Subscribers
        rospy.Subscriber('/local_cost_map_raw', OccupancyGrid, self.local_cost_map_callback)
        rospy.Subscriber('/selected_visual_prompt', Int32MultiArray, self.visual_prompt_callback)
        
        # Publisher
        self.goal_pub = rospy.Publisher('/goal', PoseStamped, queue_size=10)
        
        # Placeholder for latest local cost map
        self.local_cost_map = None

    def local_cost_map_callback(self, data):
        self.local_cost_map = data  # Store latest cost map data

    def visual_prompt_callback(self, data):
        if not self.local_cost_map:
            rospy.logwarn("No local cost map data received yet.")
            return
        
