#!/usr/bin/env python
import rospy
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseArray, Pose

class DummySubgoalSampler:
    def __init__(self, local_map_topic='/local_cost_map_raw'):
        rospy.init_node('subgoal_sampler', anonymous=False)
        self.local_map_topic = local_map_topic
        # Subscriber
        rospy.Subscriber(self.local_map_topic, OccupancyGrid, self.local_cost_map_callback)

        # Publisher
        self.goal_samples_pub = rospy.Publisher('/goal_samples', PoseArray, queue_size=10)
        
        # Main loop
        self.rate = rospy.Rate(1)
        

    def local_cost_map_callback(self, data:OccupancyGrid):
        # Placeholder for processing local cost map data
        print(f"Received local cost map data: {data.header}")
        array = PoseArray()

        for i in range(4):
            pose = Pose()
            pose.position.x = i
            array.poses.append(pose)

        self.goal_samples_pub.publish(array)
        pass

if __name__ == '__main__':
    DummySubgoalSampler('/move_base/local_costmap/costmap')
    while not rospy.is_shutdown():
            rospy.spin()
