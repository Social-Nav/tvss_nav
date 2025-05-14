import json
import os
import signal
import sys
import base64
import threading
import yaml
import time
from typing import List, Dict, Any
import roslibpy
import re
from openai import OpenAI
from tools.sfm_config.sfm_config import update_sfm_param

class VLM:
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
        print("Starting VLM node...")
        self.ros = roslibpy.Ros(host='localhost', port=9090)
        self.ros.run()
        if not self.ros.is_connected:
            print("Waiting for ROS connection...")
            while not self.ros.is_connected and not self.shutdown_flag:
                time.sleep(0.1)
            if self.ros.is_connected:
                print("✓ Connected to ROS")
            else:
                raise ConnectionError("Failed to connect to ROS")
            
        self.text_publisher = roslibpy.Topic(
            self.ros,
            '/text_input',
            'std_msgs/String'
        )
        # Load tools from YAML
        tools_path = os.path.join(os.path.dirname(__file__), 'tools.yaml')
        with open(tools_path, 'r') as f:
            self.tools = yaml.safe_load(f)['tools']
        
        # Load configuration
        config_path = os.path.join(os.path.dirname(__file__), 'config.json')
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.config['api_key'])
        
        # Load system prompt
        prompt_path = os.path.join(os.path.dirname(__file__), 'system_prompt.txt')
        with open(prompt_path, 'r') as f:
            self.system_prompt = f.read().strip()
        
        # Initialize image queue and latest image
        self.latest_image = None
        self.image_lock = threading.Lock()
        self.received_first_image = False
        
        # Subscribe to compressed image topic
        print(f"Subscribing to camera topic: /camera/color/image_raw/compressed")
        self.image_sub = roslibpy.Topic(
            self.ros,
            '/camera/color/image_raw/compressed',
            'sensor_msgs/CompressedImage'
        )
        self.image_sub.subscribe(self.image_callback)
        
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
        print("\nShutting down...")
        self.shutdown_flag = True
        if self.input_event:
            self.input_event.set()
        self.shutdown_callback()
        sys.exit(0)

    def image_callback(self, msg):
        """Callback for receiving compressed images"""
        with self.image_lock:
            self.latest_image = msg
            if not self.received_first_image:
                print("✓ Camera connection established")
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
            print("\nSending request to OpenAI API…")
        else:
            print("\nProcessing…")

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
        print(assistant_content)
        if self.debug:
            print(f"Tokens used - Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}")

        # Process any tool calls from the response
        if hasattr(response.choices[0].message, "tool_calls"):
            tool_calls = response.choices[0].message.tool_calls
            if tool_calls:
                for tool_call in tool_calls:
                    if tool_call.function.name == "segment_social_entities_from_name":
                        try:
                            params = json.loads(tool_call.function.arguments)
                            objects = params['object_names'].split('.')
                            msg = f"Segmenting: {', '.join(objects)}"
                            print(msg)
                            text_msg = {'data': params['object_names'].strip()}
                            self.text_publisher.publish(roslibpy.Message(text_msg))
                            # Here you can add the actual segmentation logic
                            # For example, calling your ROS service or handling the segmentation
                        except json.JSONDecodeError as e:
                            if self.debug:
                                print(f"Failed to parse tool arguments: {str(e)}")
                    elif tool_call.function.name == "update_sfm_param":
                        try:
                            params = json.loads(tool_call.function.arguments)
                            result = self.update_sfm_param(
                                params['param_name'],
                                float(params['value'])
                            )
                            print(result)
                        except json.JSONDecodeError as e:
                            if self.debug:
                                print(f"Failed to parse tool arguments: {str(e)}")
                        except Exception as e:
                            if self.debug:
                                print(f"Failed to update SFM parameter: {str(e)}")

        # Update message history
        current_response = {"role": "assistant", "content": assistant_content}
        self.message_history.append(user_message)
        self.message_history.append(current_response)
        if len(self.message_history) > self.max_history * 2:
            self.message_history = self.message_history[-self.max_history*2:]

        return assistant_content

    def run(self):
        """Main loop for handling user input and periodic updates"""
        print("\nVLM ready. Enter your queries (Ctrl+C to exit, press 'q' to restart):")
        try:
            while not self.shutdown_flag and self.ros.is_connected:
                if self.paused:
                    if not self.received_first_image:
                        continue
                    user_query = input("\nQuery: ").strip()
                    if user_query.lower() == 'q':
                        print("\nRestarting reasoning…")
                        self.message_history.clear()
                        self.initial_task = None  # Reset initial task
                        continue
                    if not user_query:
                        continue
                    self.paused = False
                    self.last_update_time = time.time()
                else:
                    user_input = self.get_input()
                    if user_input and user_input.lower() == 'q':
                        print("\nRestarting reasoning…")
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
                    print("Waiting for camera input…")
                    continue

                img_b64 = self.process_image(img)
                self.query_gpt(img_b64, user_query)
                print("\n" + "-"*50)

        except KeyboardInterrupt:
            print("\nShutting down…")
        finally:
            self.shutdown_flag = True
            if self.input_event:
                self.input_event.set()
            self.shutdown_callback()

def main():
    vlm = VLM(debug=False, update_period=10.0)
    vlm.run()

if __name__ == '__main__':
    main()
