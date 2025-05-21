import base64
import cv2
import json
import numpy as np
import os
import re
import signal
import sys
import threading
import time
from datetime import datetime
from typing import Any, Dict, List

import roslibpy
import yaml
from openai import OpenAI

from utils.json_parser import extract_json_from_markdown

# Set up logging
base_dir = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(base_dir, f"logs/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
os.makedirs(log_dir, exist_ok=True)
log_image_dir = os.path.join(log_dir, "images")
os.makedirs(log_image_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"subgoal_sampler_log.txt")
log_fp = open(log_file, "w")

def log(msg: str):
    print(msg)
    log_fp.write(msg + "\n")
    log_fp.flush()
    
class SubgoalSampler:
    def __init__(self, debug=False, update_period=10.0):
        # Initialize flags
        self.shutdown_flag = False
        self.debug = debug
        self.paused = True  # Start in paused state, waiting for initial query
        
        # Timer settings
        self.update_period = update_period
        self.last_update_time = 0
        
        # Context management
        self.message_history: List[Dict[str, Any]] = []
        self.max_history = 4  # Maximum number of message pairs to keep
        self.initial_task = None  # Store the initial task description
        
        # Threading control
        self.input_event = threading.Event()
        self.user_input = None
        self.input_thread = None
        
        # Setup ROS connection
        log("Starting SubgoalSampler node...")
        self.ros = roslibpy.Ros(host='localhost', port=9090)
        self.ros.run()
        if not self.ros.is_connected:
            log("Waiting for ROS connection...")
            while not self.ros.is_connected and not self.shutdown_flag:
                time.sleep(0.1)
            if self.ros.is_connected:
                log("✓ Connected to ROS")
            else:
                raise ConnectionError("Failed to connect to ROS")
        
        # Initialize sam input text publishers
        self.text_publisher = roslibpy.Topic(self.ros, '/text_input', 'std_msgs/String')
        
        # Load configuration
        config_path = os.path.join(os.path.dirname(__file__), 'config.json')
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Initialize OpenAI client
        # self.api_key = os.getenv('OPENAI_API_KEY')
        self.api_key = os.getenv('ARK_API_KEY')
        if not self.api_key:
            raise ValueError("api key environment variable not set")
        # self.client = OpenAI(api_key=self.api_key)
        self.client = OpenAI(base_url="https://ark.cn-beijing.volces.com/api/v3", api_key=self.api_key)
        
        # Load system prompt
        prompt_path = os.path.join(os.path.dirname(__file__), 'system_prompt.txt')
        with open(prompt_path, 'r') as f:
            self.system_prompt = f.read().strip()
        
        # Initialize image queue and latest image
        self.latest_image = None
        self.image_lock = threading.Lock()
        self.received_first_image = False
        
        # Subscribe to compressed image topic
        self.image_sub = roslibpy.Topic(self.ros, '/camera/color/image_raw/compressed', 'sensor_msgs/CompressedImage')
        self.image_sub.subscribe(self.image_callback)
        log("Subscribing to camera topic: /camera/color/image_raw/compressed")
        
        self.camera_info = None
        self.caminfo_sub = roslibpy.Topic(self.ros, '/camera/color/camera_info', 'sensor_msgs/CameraInfo')
        self.caminfo_sub.subscribe(self.caminfo_callback)
        log("Subscribing to camera info topic: /camera/color/camera_info")
        
        self.goal_pub = roslibpy.Topic(self.ros, '/pixel_subgoal', 'geometry_msgs/PointStamped')

        self.processing_image_timestamp = None
        self.query_counter = 0
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def shutdown_callback(self):
        """Shutdown callback"""
        self.shutdown_flag = True
        if self.input_event:
            self.input_event.set()
        if self.ros.is_connected:
            self.image_sub.unsubscribe()
            self.ros.terminate()

    def signal_handler(self, signum, frame):
        """Handle SIGINT and SIGTERM"""
        log("\nShutting down...")
        self.shutdown_flag = True
        if self.input_event:
            self.input_event.set()
        self.shutdown_callback()
        log_fp.close()
        sys.exit(0)

    def image_callback(self, msg):
        """Callback for receiving compressed images"""
        with self.image_lock:
            self.latest_image = msg
            if not self.received_first_image:
                log("✓ Camera connection established")
                self.received_first_image = True

    def caminfo_callback(self, msg):
        self.camera_info = msg
            
    def process_image(self, compressed_msg):
        """Convert compressed image message to base64"""
        self.processing_image_timestamp = compressed_msg['header']['stamp']
        
        if isinstance(compressed_msg['data'], str):
            image_base64 = compressed_msg['data']
        else:
            image_base64 = base64.b64encode(compressed_msg['data']).decode('utf-8')
        return image_base64

    def input_worker(self):
        """Thread worker for handling input"""
        try:
            self.user_input = input()
            self.input_event.set()
        except (EOFError, KeyboardInterrupt):
            self.input_event.set()

    def get_input(self, prompt=None, timeout=0.1):
        """Get input with timeout"""
        if prompt:
            print(prompt, end='', flush=True)
        self.user_input = None
        self.input_event.clear()
        self.input_thread = threading.Thread(target=self.input_worker)
        self.input_thread.daemon = True
        self.input_thread.start()
        self.input_event.wait(timeout)
        return self.user_input

    def query_gpt(self, image_base64, user_query=None):
        """Query GPT with image and user input; single call to correctly get token usage"""
        # Store initial task if this is the first query
        if not self.message_history and user_query:
            self.initial_task = user_query

        # Save image to file for debugging
        image_name = f"image_{self.query_counter}.jpg"
        image_path = os.path.join(log_image_dir, image_name)
        img_data = base64.b64decode(image_base64)
        
        with open(image_path, "wb") as img_file:
            img_file.write(img_data)
        log(f"Image saved as {image_name}")
        
        # Construct messages
        messages = [{"role": "system", "content": self.system_prompt}]
        if self.initial_task:
            messages.append({"role": "user", "content": self.initial_task})
        messages.extend(self.message_history)
        image_uri = f"data:image/jpeg;base64,{image_base64}"
        text = user_query if user_query else (
            "Now the image is more than 10s after what you last seen. Based on your observation, "
            "predict your navigation goal on the image space with [x,y] pixel coordinate."
        )

        image_width = self.camera_info["width"]
        image_height = self.camera_info["height"]

        image_size_info = f"The image resolution is {image_width}x{image_height} pixels."

        text += f"\n\n{image_size_info}"

        user_message = {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": image_uri}}
            ]
        }
        messages.append(user_message)

        if self.debug:
            log("\nSending request to OpenAI API…")
        else:
            log("\nProcessing…")

        # One-time call, no stream, to get usage
        response = self.client.chat.completions.create(
            model=self.config['model'],
            messages=messages,
            max_tokens=self.config['max_text_tokens'],
            stream=False
        )

        # Get content and token statistics
        assistant_content = response.choices[0].message.content
        usage = response.usage
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        total_tokens = usage.total_tokens

        # Output content
        log(assistant_content)
        if self.debug:
            log(f"Tokens used - Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}")

        if not assistant_content:
            log("No content in the response")
            return None
            
        json_str = extract_json_from_markdown(assistant_content)
        data = json.loads(json_str)
        coords = data["subgoal pixel coordinates"]
        pgx, pgy = coords[0], coords[1]

        # show the goal in the image
        try:
            img_array = np.frombuffer(img_data, dtype=np.uint8)
            image_cv2 = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            cv2.circle(image_cv2, (pgx, pgy), radius=6, color=(0, 0, 255), thickness=-1)

            # 显示图像
            cv2.imshow("Image with Goal", image_cv2)
            cv2.waitKey(5000)

            image_with_goal_name = f"image_{self.query_counter}_with_goal.jpg"
            image_with_goal_path = os.path.join(log_image_dir, image_with_goal_name)
            cv2.imwrite(image_with_goal_path, image_cv2)
            log(f"Saved image with goal as: {image_with_goal_name}")

        except Exception as e:
            log(f"Failed to display/save image with goal: {e}")
        
        goal_msg = {
            'header': {
                'stamp': self.processing_image_timestamp,
                'frame_id': self.camera_info['header']['frame_id']
            },
            'point': {
                'x': pgx,
                'y': pgy,
                'z': 0.0
            }
        }
        self.goal_pub.publish(roslibpy.Message(goal_msg))

        # Update message history
        current_response = {"role": "assistant", "content": assistant_content}
        self.message_history.append(user_message)
        self.message_history.append(current_response)
        if len(self.message_history) > self.max_history * 2:
            self.message_history = self.message_history[-self.max_history*2:]

        return assistant_content

    def run(self):
        """Main loop for handling user input and periodic updates"""
        log("\nSubgoalSampler ready.")
        try:
            while not self.shutdown_flag and self.ros.is_connected:
                if self.paused:
                    if not self.received_first_image:
                        continue
                    
                    log("\nEnter your queries (Ctrl+C to exit, press 'q' to restart):")
                    user_query = self.get_input("\nQuery: ", timeout=100000)
                    if not user_query:
                        continue
                    if user_query.lower() == 'q':
                        log("\nRestarting reasoning…")
                        self.message_history.clear()
                        self.initial_task = None
                        continue
                    
                    self.paused = False
                    self.last_update_time = time.time()
                else:
                    user_input = self.get_input()
                    if user_input and user_input.lower() == 'q':
                        log("\nRestarting reasoning…")
                        self.paused = True
                        self.message_history.clear()
                        self.initial_task = None  # Reset initial task
                        continue
                    if time.time() - self.last_update_time < self.update_period:
                        continue
                    user_query = None
                    self.last_update_time = time.time()

                with self.image_lock:
                    img = self.latest_image
                if img is None:
                    log("Waiting for camera input…")
                    continue

                img_b64 = self.process_image(img)
                self.query_gpt(img_b64, user_query)
                self.query_counter += 1
                log("\n" + "-"*50)

        except KeyboardInterrupt:
            log("\nShutting down…")
        finally:
            self.shutdown_flag = True
            if self.input_event:
                self.input_event.set()
            self.shutdown_callback()
            log_fp.close()

def main():
    for var in ["ALL_PROXY", "all_proxy"]:
        os.environ.pop(var, None)
        
    subgoal_sampler = SubgoalSampler(debug=False, update_period=10.0)
    subgoal_sampler.run()

if __name__ == '__main__':
    main()