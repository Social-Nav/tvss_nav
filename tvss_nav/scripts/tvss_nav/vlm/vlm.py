#!/usr/bin/env python3

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
import numpy as np
import cv2
from openai import OpenAI

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
        self.max_history = 10  # Maximum number of message pairs to keep
        
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
        
        # Load tools from YAML (instead of functions)
        tools_path = os.path.join(os.path.dirname(__file__), 'tools.yaml')
        with open(tools_path, 'r') as f:
            self.tools = yaml.safe_load(f)['tools']
        
        # Load configuration
        config_path = os.path.join(os.path.dirname(__file__), 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=config['api_key'])
        
        # Load system prompt
        prompt_path = os.path.join(os.path.dirname(__file__), 'system_prompt.txt')
        with open(prompt_path, 'r') as f:
            self.system_prompt = f.read().strip()
        
        # Store config
        self.config = config
        
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

    def query_gpt(self, image_base64, user_query=None):
        """Query GPT with image and user input, forcing tool calls"""
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.message_history)
        image_uri = f"data:image/jpeg;base64,{image_base64}"
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": user_query if user_query else "Now describe what has changed in the image. Then use your tools to segment the critical social entities."},
                {"type": "image_url", "image_url": {"url": image_uri}}
            ]
        })

        if self.debug:
            print("\nSending request to OpenAI API…")
        else:
            print("\nProcessing…")

        stream = self.client.chat.completions.create(
            model=self.config['model'],
            messages=messages,
            max_tokens=self.config['max_text_tokens'],
            tools=self.tools,
            tool_choice="auto",
            stream=True
        )

        full_response = ""
        current_response = {"role": "assistant", "content": ""}
        tool_calls_buffer = {}  # Dict to track incomplete tool calls

        for chunk in stream:
            if self.shutdown_flag:
                break
            delta = chunk.choices[0].delta

            if hasattr(delta, "content") and delta.content is not None:
                print(delta.content, end='', flush=True)
                full_response += delta.content
                current_response['content'] += delta.content

            if hasattr(delta, "tool_calls") and delta.tool_calls:
                for tc in delta.tool_calls:
                    # Get or create buffer for this tool call
                    call_id = tc.index
                    if call_id not in tool_calls_buffer:
                        tool_calls_buffer[call_id] = {
                            "name": "",
                            "arguments": "",
                            "complete": False
                        }
                    
                    # Update function name if present
                    if tc.function.name:
                        tool_calls_buffer[call_id]["name"] = tc.function.name
                    
                    # Append arguments if present
                    if tc.function.arguments:
                        tool_calls_buffer[call_id]["arguments"] += tc.function.arguments

        print()  # New line after streaming output

        # Process completed tool calls
        for call_id, call_info in tool_calls_buffer.items():
            name = call_info["name"]
            args = call_info["arguments"]
            try:
                params = json.loads(args)
                if self.debug:
                    print(f"[Tool Call] {name} → {params}")
                # Handle tool implementation
                if name == "segment_social_entities_from_name":
                    objects = params['object_names'].split('.')
                    msg = f"Segmenting: {', '.join(objects)}"
                    print(msg)
                    full_response += msg
                    current_response['content'] += msg
            except Exception as e:
                if self.debug:
                    print(f"Tool call parse error for {name}: {str(e)}")
                    print(f"Raw arguments: {args}")
                    print(f"Tool call buffer state: {tool_calls_buffer}")  # Add more debugging info

        # 更新对话历史
        if user_query:
            user_message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_query},
                    {"type": "image_url", "image_url": {"url": image_uri}}
                ]
            }
            self.message_history.append(user_message)
            self.message_history.append(current_response)
            # 保持历史长度
            if len(self.message_history) > self.max_history * 2:
                self.message_history = self.message_history[-self.max_history*2:]

        return full_response

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
    vlm = VLM(debug=True, update_period=5.0)
    vlm.run()

if __name__ == '__main__':
    main()
