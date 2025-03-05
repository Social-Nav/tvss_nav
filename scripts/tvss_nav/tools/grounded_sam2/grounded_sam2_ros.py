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
import roslibpy
from threading import Lock

# Import utilities from ros_util
from ros_utils import (
    decode_compressed_image, 
    decode_image, 
    encode_image_to_compressed, 
    create_compressed_image_message,
    setup_ros_bridge,
    create_subscriber,
    create_publisher
)

#####################
# Exposed functions for external use
#####################
def perform_initial_detection(frame, prompt, model_cfg, sam2_checkpoint, model_id, width, height):
    """
    Given an image frame and a text prompt, initialize predictor with the first frame and run detection.
    Returns:
      predictor: Initialized SAM2 predictor.
      id_to_objects: Dictionary mapping object IDs to detected class labels.
    """
    predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)
    text = prompt.lower() + "."
    frame_resized = cv2.resize(frame, (width, height))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(frame_rgb)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(model_id)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    
    inputs = processor(images=image_pil, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = grounding_model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.3,
        text_threshold=0.3,
        target_sizes=[image_pil.size[::-1]]
    )
    predictor.load_first_frame(frame_rgb)
    input_boxes = results[0]["boxes"].cpu().numpy()
    OBJECTS = results[0]["labels"]
    ann_obj_id = 1
    for box in input_boxes:
        start_pt = np.array([box[0], box[1]], dtype=np.float32)
        end_pt = np.array([box[2], box[3]], dtype=np.float32)
        bbox = np.array([start_pt, end_pt], dtype=np.float32)
        predictor.add_new_prompt(frame_idx=0, obj_id=ann_obj_id, bbox=bbox)
        ann_obj_id += 1
    id_to_objects = {i: obj for i, obj in enumerate(OBJECTS, start=1)}
    # print("Initial detection:", id_to_objects)
    return predictor, id_to_objects

def track_frame(predictor, frame, width, height):
    """
    Given an active predictor and a new frame, track detections.
    Returns:
      out_obj_ids: List of object IDs.
      out_mask_logits: Segmentation mask logits.
      frame_resized: The resized frame used for tracking.
    """
    frame_resized = cv2.resize(frame, (width, height))
    
    # Updated to handle varying return value count from predictor.track()
    result = predictor.track(frame_resized)
    
    # Check if track returns 2 or 3 values and handle accordingly
    if isinstance(result, tuple) and len(result) > 2:
        out_obj_ids, out_mask_logits = result[0], result[1]
    else:
        out_obj_ids, out_mask_logits = result
        
    return out_obj_ids, out_mask_logits, frame_resized

def visualize_detections(frame_resized, out_obj_ids, out_mask_logits, id_to_objects):
    """
    Creates a visualization overlay of detections on the given frame.
    """
    all_mask = np.zeros((frame_resized.shape[0], frame_resized.shape[1], 1), dtype=np.uint8)
    for i in range(len(out_obj_ids)):
        out_mask = (out_mask_logits[i] > 0.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8) * 255
        all_mask = cv2.bitwise_or(all_mask, out_mask)
    all_mask = cv2.cvtColor(all_mask, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(frame_resized, 1, all_mask, 0.5, 0)
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
    overlay = label_annotator.annotate(overlay, detections=detections, labels=[id_to_objects[i] for i in out_obj_ids])
    mask_annotator = sv.MaskAnnotator()
    overlay = mask_annotator.annotate(scene=overlay, detections=detections)
    return overlay

# Function moved outside of main
def input_thread_function(callback_fn):
    """
    Thread function that reads text input from user.
    """
    while True:
        text = input("Enter text prompt ('q' to quit): ")
        if text.lower() == 'q':
            print("Exiting input thread")
            break
        callback_fn(text)

#####################
# Main function using exposed functions with full functionality
#####################
def main():
    #####################
    # Configurable Parameters
    #####################
    INPUT_IMAGE_TOPIC = '/robot_firstperson_rgb/compressed'
    OUTPUT_IMAGE_TOPIC = '/segmented_image'
    IMAGE_MSG_TYPE = "CompressedImage"  # "CompressedImage" or "Image"
    ENABLE_IMAGE_PUBLISH = True
    DEBUG_MODE = False
    HEIGHT = 480
    WIDTH = 640

    # Model and checkpoint settings
    SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_tiny.pt"
    MODEL_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"
    MODEL_ID = "IDEA-Research/grounding-dino-tiny"

    #####################
    # State variables (now local to main)
    #####################
    global_frame = None
    last_image_time = None
    frame_lock = Lock()
    text_prompt = None
    restart_inference = False

    #####################
    # Image callback with closure
    #####################
    def image_callback(message):
        nonlocal global_frame, last_image_time
        with frame_lock:
            if IMAGE_MSG_TYPE == "CompressedImage":
                global_frame = decode_compressed_image(message)
            elif IMAGE_MSG_TYPE == "Image":
                global_frame = decode_image(message, HEIGHT, WIDTH)
            else:
                print("Unsupported IMAGE_MSG_TYPE:", IMAGE_MSG_TYPE)
            last_image_time = time.time()

    #####################
    # Handle text input
    #####################
    def handle_text_input(text):
        nonlocal text_prompt, restart_inference
        text_prompt = text
        restart_inference = True

    #####################
    # ROSBridge Setup
    #####################
    ros = setup_ros_bridge()
    
    msg_type = 'sensor_msgs/CompressedImage' if IMAGE_MSG_TYPE=="CompressedImage" else 'sensor_msgs/Image'
    subscriber = create_subscriber(ros, INPUT_IMAGE_TOPIC, msg_type, image_callback)
    
    publisher_compressed = create_publisher(ros, OUTPUT_IMAGE_TOPIC + '/compressed', 'sensor_msgs/CompressedImage')
    if DEBUG_MODE:
        print(f"Created publisher for topic: {OUTPUT_IMAGE_TOPIC}/compressed")

    #####################
    # Model initialization
    #####################
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Start the input thread
    input_t = threading.Thread(target=input_thread_function, args=(handle_text_input,))
    input_t.daemon = True
    input_t.start()

    # Main processing loop
    predictor = None
    id_to_objects = None
    rate = 0.1
    
    try:
        while True:
            # Check if we need to reinitialize detection with new text prompt
            if restart_inference and text_prompt:
                with frame_lock:
                    if global_frame is None:
                        continue
                    frame = global_frame.copy()
                predictor, id_to_objects = perform_initial_detection(
                    frame, text_prompt, MODEL_CFG, SAM2_CHECKPOINT, MODEL_ID, WIDTH, HEIGHT)
                restart_inference = False
                
            # If predictor isn't initialized yet, wait for text prompt
            if predictor is None:
                time.sleep(rate)
                continue
                
            # Process current frame
            with frame_lock:
                if global_frame is None:
                    continue
                frame = global_frame.copy()
            
            out_obj_ids, out_mask_logits, frame_resized = track_frame(predictor, frame, WIDTH, HEIGHT)
            overlay = visualize_detections(frame_resized, out_obj_ids, out_mask_logits, id_to_objects)
            cv2.imshow("Segmented Frame", overlay)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting main")
                break
            
            # Publish processed image if enabled
            if ENABLE_IMAGE_PUBLISH:
                try:
                    compressed_msg = create_compressed_image_message(overlay)
                    publisher_compressed.publish(roslibpy.Message(compressed_msg))
                    if DEBUG_MODE:
                        print(f"Published image with size: {overlay.shape}")
                except Exception as e:
                    print(f"Error publishing image: {e}")
            
            time.sleep(rate)
    except KeyboardInterrupt:
        print("Program interrupted")
    finally:
        # Cleanup
        cv2.destroyAllWindows()
        ros.terminate()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Exception occurred:", e)
    finally:
        cv2.destroyAllWindows()
