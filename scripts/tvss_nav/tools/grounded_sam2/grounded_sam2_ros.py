import os
import cv2
import torch
import numpy as np
import supervision as sv
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor, build_sam2, build_sam2_camera_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images
import time
import threading
import base64
import roslibpy  # Use roslibpy for ROSBridge communication
from threading import Lock

#####################
# Custom Configuration Parameters
#####################
# Input image topic name
INPUT_IMAGE_TOPIC = '/robot_firstperson_rgb/compressed'
# Output segmented image topic name (without /compressed suffix)
OUTPUT_IMAGE_TOPIC = '/segmented_image'
# Input image message type: "CompressedImage" or "Image"
IMAGE_MSG_TYPE = "CompressedImage"
# Enable/Disable image publishing
ENABLE_IMAGE_PUBLISH = True
# Enable debug messages
DEBUG_MODE = False
# Set Height and Width for the image
HEIGHT = 480
WIDTH = 640
#####################

# Global variables for storing the latest received image and its timestamp
global_frame = None
last_image_time = None  # Record the last time an image was received
last_warning_time = 0   # Record the last warning time
frame_lock = Lock()

# Global text prompt variables (input from command line)
text_prompt = None
restart_inference = False

#####################
# ROSBridge Settings
#####################
# Connect to rosbridge_server (modify host and port as needed)
ros = roslibpy.Ros(host='localhost', port=9090)
ros.run()

# Note: No longer subscribing to text prompt via ROS, using command line input

#####################
# Define image decoding functions
#####################
def decode_compressed_image(msg):
    """
    Decode sensor_msgs/CompressedImage message to OpenCV image
    msg['data'] is a base64 encoded string
    """
    img_data = base64.b64decode(msg['data'])
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img

def decode_image(msg):
    """
    Decode sensor_msgs/Image message to OpenCV image
    Assuming encoding is "bgr8"
    """
    img_data = base64.b64decode(msg['data'])
    np_arr = np.frombuffer(img_data, dtype=np.uint8)
    height = msg.get('height', HEIGHT)
    width = msg.get('width', WIDTH)
    img = np_arr.reshape((height, width, 3))
    return img

#####################
# Image callback function: Update global image and timestamp
#####################
def image_callback(message):
    global global_frame, last_image_time
    with frame_lock:
        if IMAGE_MSG_TYPE == "CompressedImage":
            global_frame = decode_compressed_image(message)
        elif IMAGE_MSG_TYPE == "Image":
            global_frame = decode_image(message)
        else:
            print("Unsupported IMAGE_MSG_TYPE:", IMAGE_MSG_TYPE)
        last_image_time = time.time()

# Subscribe to image topic
image_topic = roslibpy.Topic(
    ros, 
    INPUT_IMAGE_TOPIC, 
    'sensor_msgs/CompressedImage' if IMAGE_MSG_TYPE=="CompressedImage" else 'sensor_msgs/Image'
)
image_topic.subscribe(image_callback)

#####################
# Create publishers for both raw and compressed images
#####################
def encode_image_to_compressed(img):
    """
    Encode OpenCV image to JPEG format, then base64 encode to construct message
    """
    # Ensure the image is in BGR format
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    retval, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    return jpg_as_text

# Create publishers for both raw and compressed images
publisher_compressed = roslibpy.Topic(ros, OUTPUT_IMAGE_TOPIC + '/compressed', 'sensor_msgs/CompressedImage')

if DEBUG_MODE:
    print(f"Created publisher for topic: {OUTPUT_IMAGE_TOPIC}/compressed")

#####################
# Model initialization (similar to original code)
#####################
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

sam2_checkpoint = "./checkpoints/sam2.1_hiera_tiny.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"

# Initialize Grounding DINO model
model_id = "IDEA-Research/grounding-dino-tiny"
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(model_id)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

#####################
# Text input thread: Command line input
#####################
def input_thread():
    global text_prompt, restart_inference
    while True:
        text = input("Enter text prompt ('q' to quit): ")
        if text.lower() == 'q':
            print("Exiting input thread")
            break
        text_prompt = text
        restart_inference = True

# Start input thread
input_t = threading.Thread(target=input_thread)
input_t.daemon = True
input_t.start()

#####################
# Main processing logic: Real-time segmentation and result publishing
#####################
def main():
    global text_prompt, restart_inference, global_frame, last_image_time, last_warning_time
    predictor = None
    ID_TO_OBJECTS = {}
    rate = 0.1  # Loop interval (seconds)
    warning_threshold = 3.0  # Warning if no image received within 3 seconds

    while ros.is_connected:
        current_time = time.time()
        # Check if no image received for a long time
        if last_image_time is None or (current_time - last_image_time > warning_threshold):
            if current_time - last_warning_time > warning_threshold:
                print(f"Warning: No image messages received in {warning_threshold} seconds!")
                last_warning_time = current_time

        # If new text prompt and image available, initialize predictor
        if restart_inference and text_prompt and global_frame is not None:
            predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)
            restart_inference = False
            text = text_prompt.lower() + "."
            with frame_lock:
                frame_local = global_frame.copy()
            frame_resized = cv2.resize(frame_local, (WIDTH, HEIGHT))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(frame_rgb)

            # Use Grounding DINO to get detection boxes
            inputs = processor(images=image_pil, text=text, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = grounding_model(**inputs)
            results = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=0.25,
                text_threshold=0.3,
                target_sizes=[image_pil.size[::-1]]
            )

            # Load first frame into predictor
            predictor.load_first_frame(frame_rgb)

            input_boxes = results[0]["boxes"].cpu().numpy()
            OBJECTS = results[0]["labels"]

            ann_obj_id = 1
            for box, label in zip(input_boxes, OBJECTS):
                start_pt = np.array([box[0], box[1]], dtype=np.float32)
                end_pt = np.array([box[2], box[3]], dtype=np.float32)
                bbox = np.array([start_pt, end_pt], dtype=np.float32)
                predictor.add_new_prompt(frame_idx=0, obj_id=ann_obj_id, bbox=bbox)
                ann_obj_id += 1

            ID_TO_OBJECTS = {i: obj for i, obj in enumerate(OBJECTS, start=1)}
            # print("Starting new inference, prompt:", text)

        # If predictor not initialized or no image received, wait
        with frame_lock:
            if predictor is None or global_frame is None:
                time.sleep(rate)
                continue
            current_frame = global_frame.copy()

        frame_resized = cv2.resize(current_frame, (WIDTH, HEIGHT))

        start_frame_time = time.time()
        # Track current frame
        out_obj_ids, out_mask_logits = predictor.track(frame_resized)
        end_frame_time = time.time()
        latency = end_frame_time - start_frame_time
        fps = 1 / latency if latency > 0 else 0

        all_mask = np.zeros((frame_resized.shape[0], frame_resized.shape[1], 1), dtype=np.uint8)
        for i in range(len(out_obj_ids)):
            out_mask = (out_mask_logits[i] > 0.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8) * 255
            all_mask = cv2.bitwise_or(all_mask, out_mask)

        all_mask = cv2.cvtColor(all_mask, cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(frame_resized, 1, all_mask, 0.5, 0)

        # Use supervision library for visualization
        masks = np.stack([(out_mask_logits[i] > 0.0).cpu().numpy() for i in range(len(out_obj_ids))], axis=0)
        masks = masks.squeeze(1)
        detections = sv.Detections(
            xyxy=sv.mask_to_xyxy(masks),
            mask=masks,
            class_id=np.array(out_obj_ids, dtype=np.int32),
        )
        box_annotator = sv.BoxAnnotator()
        overlay = box_annotator.annotate(scene=overlay.copy(), detections=detections)
        label_annotator = sv.LabelAnnotator()
        overlay = label_annotator.annotate(overlay, detections=detections, labels=[ID_TO_OBJECTS[i] for i in out_obj_ids])
        mask_annotator = sv.MaskAnnotator()
        overlay = mask_annotator.annotate(scene=overlay, detections=detections)

        cv2.putText(overlay, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)
        cv2.putText(overlay, f"Latency: {latency:.4f} s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

        # Display processed result (for local debugging, optional)
        cv2.imshow("Segmented Frame", overlay)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting program")
            break

        # Publish processed image if enabled
        if ENABLE_IMAGE_PUBLISH:
            try:
                # Publish compressed image
                jpg_encoded = encode_image_to_compressed(overlay)
                compressed_msg = {
                    'header': {
                        'stamp': {'secs': int(time.time()), 'nsecs': 0},
                        'frame_id': 'camera_frame'
                    },
                    'format': 'jpeg',
                    'data': jpg_encoded
                }
                publisher_compressed.publish(roslibpy.Message(compressed_msg))
                
                if DEBUG_MODE:
                    print(f"Published image with size: {overlay.shape}")
                
            except Exception as e:
                print(f"Error publishing image: {e}")

        time.sleep(rate)

    cv2.destroyAllWindows()
    ros.terminate()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Exception occurred:", e)
