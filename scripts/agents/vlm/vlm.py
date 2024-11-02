#!/usr/bin/env python
import rospy
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseArray, Pose
from std_msgs.msg import String, Int32MultiArray
from sensor_msgs.msg import Image, CompressedImage

class DummyVLM:
    def __init__(self):
        rospy.init_node('vlm', anonymous=False)

        # Subscribers
        rospy.Subscriber('/task_description', String, self.task_description_callback)
        rospy.Subscriber('/current_observation', Image, self.current_observation_callback)
        rospy.Subscriber('/samples_pixel_coordinate', Int32MultiArray, self.samples_pixel_callback)
        
        # Publishers
        self.selected_visual_prompt_pub = rospy.Publisher('/selected_visual_prompt', String, queue_size=10)
        self.function_call_pub = rospy.Publisher('/function_call', String, queue_size=10)
        

    def task_description_callback(self, data):
        print(f"Received task description: {data}")
        self.function_call_pub.publish("function_call")
        pass

    def current_observation_callback(self, data):
        pass

    def samples_pixel_callback(self, data):
        pass

if __name__ == '__main__':
    DummyVLM()
    while not rospy.is_shutdown():
        rospy.spin()
