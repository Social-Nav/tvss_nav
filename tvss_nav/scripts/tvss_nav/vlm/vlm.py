#!/usr/bin/env python3

import rospy
import json
import os
import signal
import sys
import base64
import threading
import yaml
import time
from sensor_msgs.msg import CompressedImage
import numpy as np
import cv2
import requests
from typing import List, Dict, Any

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
        
        # Initialize function call state
        self.current_function_call = {
            "name": None,
            "arguments": ""
        }
        
        # Threading control
        self.input_event = threading.Event()
        self.user_input = None
        self.input_thread = None
        
        # Setup ROS node
        rospy.init_node('vlm_node', anonymous=True)
        rospy.on_shutdown(self.shutdown_callback)
        
        # Load functions from YAML
        functions_path = os.path.join(os.path.dirname(__file__), 'functions.yaml')
        with open(functions_path, 'r') as f:
            self.functions = yaml.safe_load(f)['functions']
        
        # Load configuration
        config_path = os.path.join(os.path.dirname(__file__), 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        # Set up headers for API requests
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config['api_key']}"
        }
        
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
        
        # Subscribe to compressed image topic and print status
        print("Starting VLM node...")
        print(f"Subscribing to camera topic: /camera/color/image_raw/compressed")
        self.image_sub = rospy.Subscriber(
            '/camera/color/image_raw/compressed',
            CompressedImage,
            self.image_callback,
            queue_size=1
        )
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def shutdown_callback(self):
        """ROS shutdown callback"""
        self.shutdown_flag = True
        if self.input_event:
            self.input_event.set()

    def signal_handler(self, signum, frame):
        """Handle SIGINT and SIGTERM"""
        print("\nShutting down...")
        self.shutdown_flag = True
        if self.input_event:
            self.input_event.set()
        rospy.signal_shutdown("User requested shutdown")
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
        image_base64 = base64.b64encode(compressed_msg.data).decode('utf-8')
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
        """Query GPT with image and user input"""
        try:
            messages = [{"role": "system", "content": self.system_prompt}]
            
            # Add message history
            messages.extend(self.message_history)
            
            # Add new query if provided
            if user_query:
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_query
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                })
            else:
                # For periodic updates, just send the new image
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What has changed in the scene?"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                })

            payload = {
                "model": self.config['model'],
                "stream": True,
                "messages": messages,
                "max_tokens": self.config['max_text_tokens'],
                "functions": self.functions,
                "function_call": "auto"
            }
            
            if self.debug:
                print("\nSending request to OpenAI API with payload:")
                print(json.dumps(payload, indent=2))
            else:
                print("\nProcessing...")
                
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=self.headers,
                json=payload,
                stream=True
            )
            
            if response.status_code != 200:
                return f"Error: Request failed with status {response.status_code}"
            
            full_response = ""
            current_response = {"role": "assistant", "content": ""}
            
            for line in response.iter_lines():
                if not line or self.shutdown_flag:
                    continue
                
                try:
                    # Skip lines that don't start with "data: "
                    line_str = line.decode('utf-8')
                    if not line_str.startswith("data: "):
                        continue
                        
                    # Remove "data: " prefix and parse JSON
                    json_str = line_str[6:]
                    if json_str.strip() == "[DONE]":
                        continue
                        
                    chunk_data = json.loads(json_str)
                    
                    if 'choices' in chunk_data:
                        delta = chunk_data['choices'][0].get('delta', {})
                        
                        # Check for function calls
                        if 'function_call' in delta:
                            fcall = delta['function_call']
                            # Update function call state
                            if 'name' in fcall:
                                self.current_function_call['name'] = fcall['name']
                            if 'arguments' in fcall:
                                self.current_function_call['arguments'] += fcall['arguments']

                            # Process complete function call
                            if self.current_function_call['name'] and self.current_function_call['arguments']:
                                try:
                                    if self.debug:
                                        print(f"\nFunction call - Name: {self.current_function_call['name']}")
                                        print(f"Arguments: {self.current_function_call['arguments']}")
                                    
                                    # Check if the arguments JSON is complete
                                    if ('{' in self.current_function_call['arguments'] and 
                                        '}' in self.current_function_call['arguments']):
                                        args = json.loads(self.current_function_call['arguments'])
                                        
                                        if self.current_function_call['name'] == 'segment_social_entities_from_name':
                                            objects = args['object_names'].split('.')
                                            segment_msg = f"\nSegmenting: {', '.join(objects)}"
                                            print(segment_msg)
                                            full_response += segment_msg
                                            current_response['content'] += segment_msg
                                        
                                        # Reset function call state
                                        self.current_function_call = {"name": None, "arguments": ""}
                                except json.JSONDecodeError:
                                    # Quietly continue accumulating if JSON is incomplete
                                    pass
                                except Exception as e:
                                    if self.debug:
                                        print(f"\nFunction call error: {str(e)}")
                                    self.current_function_call = {"name": None, "arguments": ""}
                        
                        # Handle regular content
                        elif 'content' in delta and delta['content'] is not None:
                            content = delta['content']
                            print(content, end='', flush=True)
                            full_response += content
                            current_response['content'] += content
                
                except Exception as e:
                    if self.debug:
                        print(f"\nStream processing error: {str(e)}")
                    continue
                
                if self.shutdown_flag:
                    break
            
            print()  # New line after streaming
            
            # Update message history
            if user_query:  # Only store if this was a user-initiated query
                user_message = {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_query},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                    ]
                }
                self.message_history.append(user_message)
                self.message_history.append(current_response)
                
                # Limit history size
                if len(self.message_history) > self.max_history * 2:  # *2 because each interaction has 2 messages
                    self.message_history = self.message_history[-self.max_history * 2:]
            
            return full_response
        except Exception as e:
            if self.debug:
                return f"Error: {str(e)}"
            return "An error occurred during processing"

    def run(self):
        """Main loop for handling user input and periodic updates"""
        print("\nVLM ready. Enter your queries (Ctrl+C to exit, press 'q' to restart reasoning):")
        
        try:
            while not rospy.is_shutdown() and not self.shutdown_flag:
                if self.paused:
                    if self.received_first_image:
                        user_query = input("\nQuery: ").strip()
                    else:
                        continue
                    
                    if user_query.lower() == 'q':
                        print("\nRestarting reasoning...")
                        self.message_history.clear()
                        continue
                        
                    if not user_query or self.shutdown_flag:
                        continue
                    
                    # Start continuous updates after first query
                    self.paused = False
                    self.last_update_time = time.time()
                else:
                    # Check for user input with timeout
                    user_input = self.get_input()
                    if user_input and user_input.lower() == 'q':
                        print("\nRestarting reasoning...")
                        self.paused = True
                        self.message_history.clear()
                        continue
                        
                    # If we haven't reached the update period yet, continue
                    if time.time() - self.last_update_time < self.update_period:
                        continue
                    
                    user_query = None  # No user query for periodic updates
                    self.last_update_time = time.time()
                
                # Get latest image
                with self.image_lock:
                    current_image = self.latest_image
                
                if current_image is None:
                    print("Waiting for camera input...")
                    continue
                
                # Process image and query GPT
                image_base64 = self.process_image(current_image)
                response = self.query_gpt(image_base64, user_query)
                if self.shutdown_flag:
                    break
                print("\n" + "-"*50)
                
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.shutdown_flag = True
            if self.input_event:
                self.input_event.set()
            rospy.signal_shutdown("User requested shutdown")

def main():
    # Set debug=True to enable detailed output
    vlm = VLM(debug=False, update_period=5.0)
    vlm.run()

if __name__ == '__main__':
    main()
