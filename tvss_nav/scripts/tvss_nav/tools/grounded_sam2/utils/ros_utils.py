import base64
import cv2
import numpy as np
import roslibpy
import threading
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
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def encode_image_to_compressed(img, format='jpg', quality=80):
    """
    Encode a CV2 image to a base64 string for ROS CompressedImage message.
    
    Args:
        img: OpenCV image (BGR or grayscale).
        format: Compression format ('jpg' or 'png').
        quality: JPEG quality (ignored if using PNG).
    
    Returns:
        Base64 string of compressed image.
    """
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if format == 'jpg':
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif format == 'png':
        encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
    else:
        raise ValueError(f"Unsupported format: {format}")

    retval, buffer = cv2.imencode('.'+format, img, encode_params)
    if not retval:
        raise ValueError("Image encoding failed.")

    return base64.b64encode(buffer).decode('utf-8')

def create_compressed_image_message(img, format='jpg', quality=80, timestamp=None, frame_link='camera_frame'):
    """
    Create a ROS CompressedImage message from a CV2 image.
    """
    if timestamp is None:
        secs = int(time.time())
        nsecs = int((time.time() - secs) * 1e9)
    else:
        secs = timestamp[0]
        nsecs = timestamp[1]

    img_encoded = encode_image_to_compressed(img, format, quality)

    return {
        'header': {
            'stamp': {'secs': secs, 'nsecs': nsecs},
            'frame_id': frame_link
        },
        'format': format,
        'data': img_encoded
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

def get_param(ros, name, default):
    param = roslibpy.Param(ros, name)
    result = {}
    event = threading.Event()

    def callback(value):
        result['value'] = value
        event.set()

    param.get(callback)
    event.wait(timeout=1.0)
    return result.get('value', default)