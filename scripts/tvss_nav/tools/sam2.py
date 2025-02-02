#!/usr/bin/env python
import rospy
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseArray
from std_msgs.msg import String
from sensor_msgs.msg import Image, CompressedImage

class DummySam2:
    def __init__(self):
        rospy.init_node('sam2', anonymous=False)

        # Subscribers
        rospy.Subscriber('/current_observation', Image, self.current_observation_callback)
        rospy.Subscriber('/function_call', String, self.function_call_callback)
        
        # Publishers
        self.image_masked_high_cost_pub = rospy.Publisher('/image_masked_high_cost', Image, queue_size=10)
        self.image_masked_middle_cost_pub = rospy.Publisher('/image_masked_middle_cost', Image, queue_size=10)
        self.image_masked_low_cost_pub = rospy.Publisher('/image_masked_low_cost', Image, queue_size=10)
        
        

    def current_observation_callback(self, data):
        pass

    def function_call_callback(self, data):
        pass
