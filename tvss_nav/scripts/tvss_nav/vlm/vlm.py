import base64
import json
import os
import signal
import sys
import threading
import time
from datetime import datetime
from typing import Any, Dict, List

import roslibpy
import yaml
from openai import OpenAI

from tools.sfm_config.sfm_config import update_sfm_param
from utils.json_parser import extract_json_from_markdown, parse_tool_calls

# Set up logging
base_dir = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(base_dir, f"logs/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
os.makedirs(log_dir, exist_ok=True)
log_image_dir = os.path.join(log_dir, "images")
os.makedirs(log_image_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"vlm_log.txt")
log_fp = open(log_file, "w")

def log(msg: str):
    print(msg)
    log_fp.write(msg + "\n")
    log_fp.flush()

class VLM:
    def __init__(self, debug=False, update_period=10.0):
        self.shutdown_flag = False
        self.debug = debug
        self.paused = True
        self.update_period = update_period
        self.last_update_time = 0

        self.message_history: List[Dict[str, Any]] = []
        self.max_history = 4
        self.initial_task = None

        self.input_event = threading.Event()
        self.user_input = None
        self.input_thread = None

        log("Starting VLM node...")
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

        # ==== Read topic names from ROS params ====
        self.camera_topic = self.get_param('/color_topic', '/camera/color/image_raw/compressed')
        self.text_topic = self.get_param('/vlm/text_input_topic', '/text_input')
        self.cost_attribute_topic = self.get_param('/vlm/cost_attribute_topic', '/cost_attributes')

        # === Publishers ===
        self.text_publisher = roslibpy.Topic(self.ros, self.text_topic, 'std_msgs/String')
        self.cost_attr_publisher = roslibpy.Topic(self.ros, self.cost_attribute_topic, 'std_msgs/String')

        # === Load tools ===
        tools_path = os.path.join(os.path.dirname(__file__), 'tools.yaml')
        with open(tools_path, 'r') as f:
            self.tools = yaml.safe_load(f)['tools']

        # === Load config.json ===
        config_path = os.path.join(os.path.dirname(__file__), 'config.json')
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # === Load system prompt ===
        prompt_path = os.path.join(os.path.dirname(__file__), 'system_prompt.txt')
        with open(prompt_path, 'r') as f:
            self.system_prompt = f.read().strip()

        # === OpenAI client ===
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("api key environment variable not set")
        self.client = OpenAI(api_key=self.api_key)

        # === Image state ===
        self.latest_image = None
        self.image_lock = threading.Lock()
        self.received_first_image = False

        self.image_sub = roslibpy.Topic(self.ros, self.camera_topic, 'sensor_msgs/CompressedImage')
        self.image_sub.subscribe(self.image_callback)
        log(f"Subscribing to camera topic: {self.camera_topic}")

        self.query_counter = 0

        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def get_param(self, name, default):
        param = roslibpy.Param(self.ros, name)
        result = {}
        event = threading.Event()

        def callback(value):
            result['value'] = value
            event.set()

        param.get(callback)
        event.wait(timeout=1.0)
        return result.get('value', default)

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

    def process_image(self, compressed_msg):
        """Convert compressed image message to base64"""
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

    def update_sfm_param(self, param_name: str, value: float) -> str:
        """Update a single SFM parameter using dynamic reconfigure.
        
        Args:
            param_name: Name of the parameter to update
            value: New value for the parameter
            
        Returns:
            Response message indicating success or failure
        """
        try:
            success = update_sfm_param(param_name, value, self.ros)
            if success:
                return f"Successfully updated {param_name} to {value}"
            return f"Failed to update {param_name}"
        except Exception as e:
            return f"Error updating parameter: {str(e)}"

    def query_gpt(self, image_base64, user_query=None):
        """Query GPT with image and user input; single call to correctly get token usage"""
        # Store initial task if this is the first query
        if not self.message_history and user_query:
            self.initial_task = user_query

        # Save image to file for debugging
        image_name = f"image_{self.query_counter}.jpg"
        image_path = os.path.join(log_image_dir, image_name)
        with open(image_path, "wb") as img_file:
            img_file.write(base64.b64decode(image_base64))
        log(f"Image saved as {image_name}")
        
        # Construct messages
        messages = [{"role": "system", "content": self.system_prompt}]
        if self.initial_task:
            messages.append({"role": "user", "content": self.initial_task})
        messages.extend(self.message_history)
        image_uri = f"data:image/jpeg;base64,{image_base64}"
        text = user_query if user_query else "Now the image is 10s after what you last seen. Based on your observation, call tools to segment all the important social entities if any and update sfm parameters if necessary."
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
            tools=self.tools,
            tool_choice="auto",
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
        # print("json_str: ", json_str)
        # Parse tool calls
        tool_calls = parse_tool_calls(json_str)
        if tool_calls:
            try:
                for tool_call in tool_calls:
                    if tool_call['tool'] == "segment_social_entities_from_name":
                        params = tool_call['args']
                        object_names = params['object_names']
                        objects = object_names.split('.')

                        # Publish object names to segment
                        msg = f"Segmenting: {', '.join(objects)}"
                        log(msg)
                        text_msg = {'data': object_names.strip()}
                        self.text_publisher.publish(roslibpy.Message(text_msg))

                        # Extract and publish cost_attributes
                        cost_attrs = params.get('cost_attributes', {})

                        # Ensure every segmented object has explicitly defined cost attributes
                        for obj in objects:
                            if obj not in cost_attrs:
                                log(f"[WARN] Missing cost attributes for object: {obj}")

                        try:
                            # Convert cost_attributes dictionary to JSON string
                            cost_json = json.dumps(cost_attrs)
                            self.cost_attr_publisher.publish(roslibpy.Message({'data': cost_json}))
                            log(f"[INFO] Published cost attributes for {len(cost_attrs)} object(s)")

                        except Exception as e:
                            print(f"[ERROR] Failed to publish cost attributes: {e}")

                        # Here you can add the actual segmentation logic
                        # For example, calling your ROS service or handling the segmentation
                        
                    elif tool_call['tool'] == "update_sfm_param":
                        params = tool_call['args']
                        result = self.update_sfm_param(
                            params['param_name'],
                            float(params['value'])
                        )
                        log(result)

            except Exception as e:
                if self.debug:
                    log(f"Failed to process tool calls: {str(e)}")

        # Update message history
        current_response = {"role": "assistant", "content": assistant_content}
        self.message_history.append(user_message)
        self.message_history.append(current_response)
        if len(self.message_history) > self.max_history * 2:
            self.message_history = self.message_history[-self.max_history*2:]

        return assistant_content

    def run(self):
        """Main loop for handling user input and periodic updates"""
        log("\nVLM ready.")
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
        
    vlm = VLM(debug=False, update_period=10.0)
    vlm.run()

if __name__ == '__main__':
    main()
