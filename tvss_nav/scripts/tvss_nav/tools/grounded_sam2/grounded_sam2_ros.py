import json
import os
import time
from threading import Lock

import cv2
import numpy as np
import roslibpy
import supervision as sv
import threading
import torch
from PIL import Image
from omegaconf import OmegaConf
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from sam2.build_sam import build_sam2, build_sam2_camera_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from utils.mask_dictionary_model import MaskDictionaryModel, ObjectInfo

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*?.*")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Import utilities from ros_util
from utils.ros_utils import (
    decode_compressed_image, 
    decode_image, 
    encode_image_to_compressed, 
    create_compressed_image_message,
    setup_ros_bridge,
    create_subscriber,
    create_publisher
)

class NoObjectDetected(Exception):
    pass

#####################
# Exposed functions for external use
#####################
def perform_init_detection(processor, grounding_model, camera_predictor, image_predictor, frame, prompt, width, height):
    """
    Given an image frame and a text prompt, initialize predictor with the first frame and run detection.
    Returns:
      predictor: Initialized SAM2 predictor.
      id_to_objects: Dictionary mapping object IDs to detected class labels.
    """
    text = prompt.lower() + "."
    frame_resized = cv2.resize(frame, (width, height))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(frame_rgb)

    inputs = processor(images=image_pil, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = grounding_model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.4,
        text_threshold=0.4,
        target_sizes=[image_pil.size[::-1]]
    )

    if len(results[0]["boxes"]) == 0:
        raise NoObjectDetected("No objects detected in the image.")
    
    image_predictor.set_image(np.array(image_pil.convert("RGB")))

    # Prepare predictor
    camera_predictor.load_first_frame(frame_rgb)
    frame_id = 0

    input_boxes = results[0]["boxes"].cpu().numpy()
    OBJECTS = results[0]["labels"]

    # prompt SAM 2 image predictor to get the mask for the object
    masks, scores, logits = image_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )
    # convert the mask shape to (n, H, W)
    if masks.ndim == 2:
        masks = masks[None]
        scores = scores[None]
        logits = logits[None]
    elif masks.ndim == 4:
        masks = masks.squeeze(1)

    mask_dict = MaskDictionaryModel(promote_type="mask", mask_name="0", mask_height=height, mask_width=width)
    # If you are using point prompts, we uniformly sample positive points based on the mask
    if mask_dict.promote_type == "mask":
        mask_dict.add_new_frame_annotation(mask_list=torch.tensor(masks).to(device), box_list=input_boxes.tolist(), label_list=OBJECTS, background_value=0)
    else:
        raise NotImplementedError("")

    id_to_objects = {}
    for object_id, object_info in mask_dict.labels.items():
        start_pt = np.array([object_info.x1, object_info.y1], dtype=np.float32)
        end_pt = np.array([object_info.x2, object_info.y2], dtype=np.float32)
        bbox = np.array([start_pt, end_pt], dtype=np.float32)
        camera_predictor.add_new_prompt(frame_idx=0, obj_id=object_id, bbox=bbox)

        id_to_objects[object_id] = object_info.class_name
    
    return id_to_objects, mask_dict

def perform_detection(processor, grounding_model, camera_predictor, image_predictor, frame, prompt, global_mask, width, height, objects_count=0):
    """
    Run detection and update predictor + global id_to_objects.
    
    Args:
        processor, grounding_model: Grounded-SAM pipeline
        predictor: SAM2CameraPredictor
        frame: current image (BGR)
        prompt: user prompt, e.g., "person"
        width, height: resize target
        id_to_objects: external global dict {obj_id: label} to be updated

    Returns:
        start_obj_id: The first obj_id assigned in this detection
        num_new: Number of objects detected
    """
    text = prompt.lower().strip() + "."
    frame_resized = cv2.resize(frame, (width, height))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(frame_rgb)

    inputs = processor(images=image_pil, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = grounding_model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.4,
        text_threshold=0.4,
        target_sizes=[image_pil.size[::-1]]
    )
    if len(results[0]["boxes"]) == 0:
        raise NoObjectDetected("No objects detected in the image.")
    
    image_predictor.set_image(np.array(image_pil.convert("RGB")))

    # Prepare predictor
    camera_predictor.add_conditioning_frame(frame_rgb)
    frame_id = camera_predictor.condition_state["num_frames"] - 1

    input_boxes = results[0]["boxes"].cpu().numpy()
    OBJECTS = results[0]["labels"]

    # prompt SAM 2 image predictor to get the mask for the object
    masks, scores, logits = image_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )
    # convert the mask shape to (n, H, W)
    if masks.ndim == 2:
        masks = masks[None]
        scores = scores[None]
        logits = logits[None]
    elif masks.ndim == 4:
        masks = masks.squeeze(1)

    mask_dict = MaskDictionaryModel(promote_type="mask", mask_name=f"{frame_id}", mask_height=height, mask_width=width)
    # If you are using point prompts, we uniformly sample positive points based on the mask
    if mask_dict.promote_type == "mask":
        mask_dict.add_new_frame_annotation(mask_list=torch.tensor(masks).to(device), box_list=input_boxes.tolist(), label_list=OBJECTS, background_value=0)
    else:
        raise NotImplementedError("")
    
    objects_count = global_mask.get_max_instance_id()
    objects_count = mask_dict.update_masks(tracking_annotation_dict=global_mask, iou_threshold=0.7, objects_count=objects_count)

    id_to_objects = {}
    for object_id, object_info in mask_dict.labels.items():
        start_pt = np.array([object_info.x1, object_info.y1], dtype=np.float32)
        end_pt = np.array([object_info.x2, object_info.y2], dtype=np.float32)
        bbox = np.array([start_pt, end_pt], dtype=np.float32)
        camera_predictor.add_new_prompt(frame_idx=frame_id, obj_id=object_id, bbox=bbox)

        id_to_objects[object_id] = object_info.class_name
    
    return id_to_objects, mask_dict

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
    all_mask = np.zeros((frame_resized.shape[0], frame_resized.shape[1], 1), dtype=np.uint8)    # HxWx1
    for i in range(len(out_obj_ids)):
        out_mask = (out_mask_logits[i] > 0.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8) * 255
        all_mask = cv2.bitwise_or(all_mask, out_mask)   
    all_mask = cv2.cvtColor(all_mask, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(frame_resized, 1, all_mask, 0.5, 0)
    masks = np.stack([(out_mask_logits[i] > 0.0).cpu().numpy() for i in range(len(out_obj_ids))], axis=0)   # NxHxW
    masks = masks.squeeze(1)
    detections = sv.Detections(
        xyxy=sv.mask_to_xyxy(masks),
        mask=masks,
        class_id=np.array(out_obj_ids, dtype=np.int32),
    )
    box_annotator = sv.BoxAnnotator()
    overlay = box_annotator.annotate(scene=overlay.copy(), detections=detections)
    label_annotator = sv.LabelAnnotator()
    overlay = label_annotator.annotate(overlay, detections=detections, labels=[f"{obj_id}: {id_to_objects[obj_id]}" for obj_id in out_obj_ids])
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
    # INPUT_IMAGE_TOPIC = '/robot_firstperson_rgb/compressed'
    # INPUT_IMAGE_TOPIC = '/camera/color/image_raw'
    INPUT_IMAGE_TOPIC = '/camera/color/image_raw/compressed'
    OUTPUT_IMAGE_TOPIC = '/segmented_image'
    IMAGE_MSG_TYPE = "CompressedImage"  # "CompressedImage" or "Image"
    # IMAGE_MSG_TYPE = "Image"
    RESET_TOPIC = '/scenario_reset'

    ENABLE_IMAGE_PUBLISH = True
    DEBUG_MODE = False
    HEIGHT = 480
    WIDTH = 640

    # Model and checkpoint settings

    SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
    MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
    MODEL_ID = "IDEA-Research/grounding-dino-base"

    model_name = os.path.splitext(os.path.basename(MODEL_CFG))[0]  # sam2.1_hiera_large
    engine_name = model_name.replace("sam2.1_", "") + "_image_encoder.trt"
    os.environ["SAM2_TRT_ENGINE_PATH"] = os.path.join(os.environ["PWD"], "tensorrt", "trt", engine_name)

    cfg = OmegaConf.load("sam2/" + MODEL_CFG)
    use_trt = cfg.model.get("use_trt", None)
    print(f"[Config] use_trt: {use_trt}")

    #####################
    # State variables (now local to main)
    #####################
    global_msg = None
    msg_lock = Lock()
    text_prompt = None
    restart_detection = False
    global_reset_signal = False

    def image_msg_parser(msg):
        if IMAGE_MSG_TYPE == "CompressedImage":
            frame = decode_compressed_image(msg)
        elif IMAGE_MSG_TYPE == "Image":
            frame = decode_image(msg, HEIGHT, WIDTH)
        else:
            print("Unsupported IMAGE_MSG_TYPE:", IMAGE_MSG_TYPE)
        frame_ts = [msg["header"]["stamp"]['secs'], msg["header"]["stamp"]['nsecs']]
        frame_link = msg["header"]["frame_id"]

        return frame, frame_ts, frame_link

    #####################
    # Image callback with closure
    #####################
    def image_callback(msg):
        nonlocal global_msg
        with msg_lock:
            global_msg = msg

    #####################
    # Handle text input
    #####################
    def handle_text_input(text):
        nonlocal text_prompt, restart_detection
        text_prompt = text
        restart_detection = True

    def task_reset_signal(msg):
        nonlocal global_reset_signal
        global_reset_signal = True

    #####################
    # ROSBridge Setup
    #####################
    ros = setup_ros_bridge()
    rgb_msg_type = 'sensor_msgs/CompressedImage' if IMAGE_MSG_TYPE=="CompressedImage" else 'sensor_msgs/Image'
    rgb_subscriber = create_subscriber(ros, INPUT_IMAGE_TOPIC, rgb_msg_type, image_callback)
    reset_subscriber = create_subscriber(ros, RESET_TOPIC, 'std_msgs/Int16', task_reset_signal)
    
    publisher_compressed = create_publisher(ros, OUTPUT_IMAGE_TOPIC + '/compressed', 'sensor_msgs/CompressedImage')
    publisher_mask = create_publisher(ros, OUTPUT_IMAGE_TOPIC + '/mask', 'sensor_msgs/CompressedImage')
    if DEBUG_MODE:
        print(f"Created publisher for topic: {OUTPUT_IMAGE_TOPIC}/compressed")

    #####################
    # Model initialization
    #####################
    torch.autocast(device_type=device, dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Start the input thread
    input_t = threading.Thread(target=input_thread_function, args=(handle_text_input,))
    input_t.daemon = True
    input_t.start()

    camera_predictor = build_sam2_camera_predictor(MODEL_CFG, SAM2_CHECKPOINT)
    sam2_image_model = build_sam2(MODEL_CFG, SAM2_CHECKPOINT, device=device)
    image_predictor = SAM2ImagePredictor(sam2_image_model)

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(device)

    id_to_objects = None
    sam2_masks = MaskDictionaryModel()

    rate = 0.1
    detection_timeout = 3000

    last_detect_time = time.time()

    initialized = False

    # text_prompt = "person"

    try:
        while True:
            if global_reset_signal:
                initialized = False
                global_reset_signal = False
                restart_detection = True
                if DEBUG_MODE:
                    print("\n[INFO] Reset signal received.")

            # Process current frame
            with msg_lock:
                if global_msg is None:
                    # print("No image message received yet.")
                    continue
                locked_frame, locked_frame_ts, locked_frame_link = image_msg_parser(global_msg)

            if restart_detection and text_prompt:
                try:
                    if not initialized:
                        if DEBUG_MODE:
                            print(f"\n[INFO] Initializing detection with prompt: {text_prompt}")

                        id_to_objects, mask_dict = perform_init_detection(
                            processor, grounding_model, camera_predictor, image_predictor, locked_frame, text_prompt, WIDTH, HEIGHT)
                        sam2_masks = mask_dict

                        initialized = True
                    else:
                        try:
                            out_obj_ids, out_mask_logits, frame_resized = track_frame(camera_predictor, locked_frame, WIDTH, HEIGHT)
                            for obj_id, mask_logit in zip(out_obj_ids, out_mask_logits):
                                mask_binary = (mask_logit > 0.0).squeeze(0)
                                if obj_id in sam2_masks.labels and mask_binary.sum() > 0:
                                    obj_info = sam2_masks.labels[obj_id]
                                    obj_info.mask = mask_binary
                                    obj_info.update_box()
                        except Exception as e:
                            print(f"\n[Warning] All tracking lost.")
                            # print(f"\n[Error] {e}")
                            pass
                        
                        camera_predictor.reset_state()
                        id_to_objects, mask_dict = perform_detection(
                            processor, grounding_model, camera_predictor, image_predictor, locked_frame, text_prompt, sam2_masks, WIDTH, HEIGHT)
                        # Appending Operation
                        for obj_id, object_info in mask_dict.labels.items():
                            if obj_id in sam2_masks.labels:
                                sam2_masks.labels[obj_id].mask = object_info.mask
                                sam2_masks.labels[obj_id].update_box()
                            if obj_id not in sam2_masks.labels:
                                sam2_masks.labels[obj_id] = object_info
                                sam2_masks.labels[obj_id].instance_id = obj_id
                                sam2_masks.labels[obj_id].update_box()

                    restart_detection = False
                except NoObjectDetected as e:
                    # print(f"\n[Warning] {e}")
                    continue
                except Exception as e:
                    print(f"\n[Error] Detection failed: {e}")
                    continue

            if not initialized:
                time.sleep(rate)
                continue

            now = time.time()
            if now - last_detect_time > detection_timeout:
                if DEBUG_MODE:
                    print(f"\n[INFO] Inference interval exceeded {detection_timeout} seconds. Reinitializing inference ...")

                restart_detection = True
                last_detect_time = now

            out_obj_ids, out_mask_logits, frame_resized = track_frame(camera_predictor, locked_frame, WIDTH, HEIGHT)
            for obj_id, mask_logit in zip(out_obj_ids, out_mask_logits):
                # Threshold the mask_logit to get a binary mask (0 or 1)
                mask_binary = (mask_logit > 0.0).squeeze(0)

                # Check if this object ID exists in the model's labels
                if obj_id in sam2_masks.labels and mask_binary.sum() > 0:
                    obj_info = sam2_masks.labels[obj_id]
                    # Update the mask of the corresponding object
                    obj_info.mask = mask_binary  # Set the binary mask
                    # Optionally update other fields like bounding box if required
                    obj_info.update_box()

            overlay = visualize_detections(frame_resized, out_obj_ids, out_mask_logits, id_to_objects)
            publish_mask(publisher_mask, frame_resized, out_obj_ids, out_mask_logits, locked_frame_ts, locked_frame_link)
            cv2.imshow("Segmented Frame", overlay)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting main")
                break
            
            # Publish processed image if enabledinstance_id
            if ENABLE_IMAGE_PUBLISH:
                try:
                    compressed_msg = create_compressed_image_message(
                        overlay, format='jpg', quality=80, timestamp=locked_frame_ts, frame_link=locked_frame_link)
                    publisher_compressed.publish(roslibpy.Message(compressed_msg))
                    if DEBUG_MODE:
                        print(f"Published image with size: {overlay.shape}")
                except Exception as e:
                    print(f"\nError publishing image: {e}")

            time.sleep(rate)
    except KeyboardInterrupt:
        print("Program interrupted")
    finally:
        # Cleanup
        cv2.destroyAllWindows()
        ros.terminate()

def publish_mask(mask_publisher, frame_resized, out_obj_ids, out_mask_logits, timestamp, frame_link):
    """ 
    Publish a Mask image with instance IDs.

    - The “mask“ values are no longer 0/255, but instead instance_id itself.
    - Uses PNG compression to reduce bandwidth.
    - Adds erosion to clean mask edges.
    """

    all_mask = np.zeros((frame_resized.shape[0], frame_resized.shape[1]), dtype=np.uint8)  # HxW

    for i, obj_id in enumerate(out_obj_ids):
        out_mask = (out_mask_logits[i] > 0.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        if out_mask.ndim == 3 and out_mask.shape[-1] == 1:
            out_mask = out_mask[..., 0]
        elif out_mask.ndim == 3:
            raise ValueError(f"Expected out_mask to have shape (H, W, 1) or (H, W), but got {out_mask.shape}")

        all_mask[out_mask > 0] = obj_id

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    all_mask = cv2.erode(all_mask, kernel, iterations=1)

    mask_msg = create_compressed_image_message(all_mask, format='png', quality=3, timestamp=timestamp, frame_link=frame_link)

    mask_publisher.publish(roslibpy.Message(mask_msg))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Exception occurred:", e)
    finally:
        cv2.destroyAllWindows()
