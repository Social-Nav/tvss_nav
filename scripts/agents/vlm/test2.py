import rospy
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage
import base64
import cv2
import numpy as np
from cv_bridge import CvBridgeError
import ollama

class VLM:
    def __init__(self, model_name="llava:latest"):
        """
        Initialize the VLM class with a specific Ollama model name.
        
        Args:
            model_name (str): The name of the Ollama model to be used.
        """
        # Initialize ROS node
        rospy.init_node('vlm_node', anonymous=False)

        # Subscribe to the compressed image topic
        rospy.Subscriber('/robot_firstperson_rgb/compressed', CompressedImage, self.image_prompt_callback)

        # Publisher for the model's response
        self.response_pub = rospy.Publisher('/vlm/response', String, queue_size=10)

        # Fixed system prompt
        self.system_prompt = (
            "You are a social robot navigating in human environments. Your mission is to follow social norms and rules, "
            "including but not limited to:\n- Yielding to pedestrians\n- Staying on sidewalks\n- Avoiding open doors\n"
            "- Avoiding vehicles\n- Using crosswalks when crossing roads\n- Keeping to the right side of the path\n"
            "- Avoiding obstacles such as chairs, tables, and other objects\n- Not entering restricted areas\n"
            "- Being polite and not startling people\n- Following traffic signals\n- Observing and obeying signs and signals\n"
            "- Maintaining a safe distance from others\n- Not blocking pathways or exits\n- Respecting personal space\n"
            "- Avoiding walls and static structures unless they block the path.\n\n"
            "Every 5 seconds, you receive an observation image from your first-person view. For each observation:\n"
            "1. Describe the scene in the image, focusing on elements relevant to social navigation (3-6 sentences).\n"
            "2. Identify relevant objects for social navigation such as people, crosswalks, doors, vehicles, signs, etc.\n"
            "3. Return the list of object names in a comma-separated format."
        )

        self.model_name = model_name
        self.processing = False  # Flag to ensure only one request at a time
        self.init_conversation()  # Initialize conversation with the system prompt

    def init_conversation(self):
        """
        Initialize the conversation by sending the system prompt once.
        """
        try:
            # Send system prompt as an initial message
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "system", "content": self.system_prompt}],
                stream=False
            )
            rospy.loginfo("System prompt initialized.")
        except ollama.ResponseError as e:
            rospy.logerr(f"Failed to initialize conversation with system prompt: {e}")

    def image_prompt_callback(self, msg):
        """
        Callback to receive the compressed image message, decode it, and generate a response if not currently processing.

        Args:
            msg (sensor_msgs.msg.CompressedImage): The compressed image message.
        """
        print("Image received")
        if self.processing:
            rospy.loginfo("Processing in progress; new request skipped.")
            return

        self.processing = True  # Set flag to prevent additional calls
        try:
            # Decode the compressed image data
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # Encode the image to Base64 format
            _, buffer = cv2.imencode('.jpg', cv_image)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            rospy.loginfo("Image received and encoded for model input.")

            # Prepare user input and make API request
            self.generate_response(image_base64)
        except CvBridgeError as e:
            rospy.logerr(f"Failed to convert image: {e}")
            self.processing = False  # Reset flag if there's an error
        except Exception as e:
            rospy.logerr(f"Unexpected error: {e}")
            self.processing = False  # Catch other exceptions

    def generate_response(self, image_base64):
        """
        Generate a response from the Ollama model based on the user's image input.

        Args:
            image_base64 (str): The Base64-encoded image.
        """
        try:
            # Use ollama library to generate response
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": f"Image Prompt (Base64): {image_base64}"}
                ],
                stream=False
            )

            # Extract the response content
            response_text = response['message']['content']
            self.publish_response(response_text)
        except ollama.ResponseError as e:
            rospy.logerr(f"Ollama API error: {e}")
        finally:
            # Reset flag after processing
            self.processing = False

    def publish_response(self, response):
        """
        Publish the model's response to a ROS topic.

        Args:
            response (str): The response generated by the model.
        """
        rospy.loginfo(f"Publishing response: {response}")
        self.response_pub.publish(String(data=response))

    def start(self):
        """
        Start the ROS node to keep it running and listening for prompts.
        """
        rospy.loginfo("Starting VLM node...")
        rospy.spin()  # Keep the node active

# Usage example
if __name__ == "__main__":
    try:
        # Initialize VLM with a default model
        vlm = VLM(model_name="llava:latest")
        
        # Start the node
        vlm.start()
    except rospy.ROSInterruptException:
        pass
