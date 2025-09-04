#!/usr/bin/env python3
import rospy
from std_msgs.msg import String

if __name__ == "__main__":
    rospy.init_node('user_query_publisher', anonymous=False)
    pub = rospy.Publisher('/user_query', String, queue_size=1, latch=True)
    rate = rospy.Rate(0.1)

    max_count = 0     
    count = 0

    while not rospy.is_shutdown():
        if count >= max_count:
            rospy.loginfo(f"Reached {max_count} publishes, shutting down.")
            break
        pub.publish(String(data="Just go with the person you see, if there is more than one people here, just randomly pick one and follow them. Your answer format should follow the above or the following prompts and system instructions."))
        rospy.loginfo(f"Published ({count+1}/{max_count}) to /user_query → 'Just go with the person you see, if there is more than one people here, just randomly pick one and follow them.'")
        count += 1
        rate.sleep()

    rospy.signal_shutdown("Completed max publishes")
