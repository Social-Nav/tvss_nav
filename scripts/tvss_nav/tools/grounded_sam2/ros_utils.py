import cv2
import numpy as np
import base64
import roslibpy
import time

def decode_compressed_image(msg):
    """
    Decode a ROS CompressedImage message to a CV2 image.
    """
    img_data = base64.b64decode(msg['data'])
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img

def decode_image(msg, height=480, width=640):
    """
    Decode a ROS Image message to a CV2 image.
    """
    img_data = base64.b64decode(msg['data'])
    np_arr = np.frombuffer(img_data, dtype=np.uint8)
    height_actual = msg.get('height', height)
    width_actual = msg.get('width', width)
    img = np_arr.reshape((height_actual, width_actual, 3))
    return img

def encode_image_to_compressed(img):
    """
    Encode a CV2 image to a base64 string for ROS CompressedImage message.
    """
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    retval, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    return jpg_as_text

def create_compressed_image_message(img, frame_id='camera_frame'):
    """
    Create a ROS CompressedImage message from a CV2 image.
    """
    jpg_encoded = encode_image_to_compressed(img)
    return {
        'header': {
            'stamp': {'secs': int(time.time()), 'nsecs': 0},
            'frame_id': frame_id
        },
        'format': 'jpeg',
        'data': jpg_encoded
    }

def setup_ros_bridge(host='localhost', port=9090):
    """
    Set up and connect to ROS Bridge.
    """
    ros = roslibpy.Ros(host=host, port=port)
    ros.run()
    return ros

def create_subscriber(ros, topic_name, msg_type, callback):
    """
    Create and return a ROS subscriber.
    """
    topic = roslibpy.Topic(ros, topic_name, msg_type)
    topic.subscribe(callback)
    return topic

def create_publisher(ros, topic_name, msg_type):
    """
    Create and return a ROS publisher.
    """
    return roslibpy.Topic(ros, topic_name, msg_type)
