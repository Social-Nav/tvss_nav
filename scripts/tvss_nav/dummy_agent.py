import rospy
import random
import math
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import threading

class DummyAgent:
    def __init__(self):
        rospy.init_node("dummy_agent", anonymous=False)

        self.goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)

        self.initial_pose = None
        rospy.Subscriber('/robot_odom', Odometry, self._odom_callback)
        rospy.loginfo("Waiting for odom to initialize...")

        while self.initial_pose is None and not rospy.is_shutdown():
            rospy.sleep(0.1)

        rospy.loginfo(f"Initial position acquired: {self.initial_pose}")

        self.goal_thread = threading.Thread(target=self._goal_generation_loop)
        self.goal_thread.daemon = True
        self.goal_thread.start()

    def _odom_callback(self, msg):
        if self.initial_pose is None:
            self.initial_pose = msg.pose.pose.position

    def _goal_generation_loop(self):
        rate = rospy.Rate(0.1)
        while not rospy.is_shutdown():
            if self.initial_pose:
                goal = self._generate_random_goal()
                self.goal_pub.publish(goal)
                rospy.loginfo(f"Published goal: {goal}")
            rate.sleep()

    def _generate_random_goal(self):
        angle = random.uniform(0, 2 * math.pi)
        radius = random.uniform(0, 10.0)

        goal_x = self.initial_pose.x + radius * math.cos(angle)
        goal_y = self.initial_pose.y + radius * math.sin(angle)

        goal = PoseStamped()
        goal.header.stamp = rospy.Time.now()
        goal.header.frame_id = "map"
        goal.pose.position.x = goal_x
        goal.pose.position.y = goal_y
        goal.pose.position.z = 0.0
        goal.pose.orientation.w = 1.0

        return goal

if __name__ == "__main__":
    agent = DummyAgent()
    rospy.spin()
