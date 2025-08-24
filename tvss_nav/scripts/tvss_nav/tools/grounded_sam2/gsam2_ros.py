import json
import os
import time
from threading import Lock
import cv2
import numpy as np
import roslibpy
import supervision as sv
import torch
from PIL import Image
from omegaconf import OmegaConf
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from sam2.build_sam import build_sam2, build_sam2_camera_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from utils.mask_dictionary_model import MaskDictionaryModel, ObjectInfo

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*?.*")

import os, logging
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")        
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1") 
logging.basicConfig(level=logging.INFO)
logging.getLogger("huggingface_hub").setLevel(logging.DEBUG)


device = "cuda" if torch.cuda.is_available() else "cpu"

# Import utilities from ros_util
from utils.ros_utils import (
    decode_compressed_image, 
    decode_image, 
    encode_image_to_compressed, 
    create_compressed_image_message,
    setup_ros_bridge,
    create_subscriber,
    create_publisher,
    get_ros_param
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
    print("[GSAM2] Performing initial detection with prompt:", prompt)
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
        box_threshold=0.1,
        text_threshold=0.1,
        target_sizes=[image_pil.size[::-1]]
    )
    print(results)
    if len(results[0]["boxes"]) == 0:
        raise NoObjectDetected("No objects detected in the image.")
    
    image_predictor.set_image(np.array(image_pil.convert("RGB")))

    # Prepare predictor
    camera_predictor.load_first_frame(frame_rgb)
    frame_id = 0

    input_boxes = results[0]["boxes"].cpu().numpy()
    OBJECTS = results[0]["labels"]
    # print("2222222222222222222222222222222222222222222222")
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
    # print("3333333333333333333333333333333333333333333333333333")
    mask_dict = MaskDictionaryModel(promote_type="mask", mask_name="0", mask_height=height, mask_width=width)
    # If you are using point prompts, we uniformly sample positive points based on the mask
    if mask_dict.promote_type == "mask":
        mask_dict.add_new_frame_annotation(mask_list=torch.tensor(masks).to(device), box_list=input_boxes.tolist(), label_list=OBJECTS, background_value=0)
    else:
        raise NotImplementedError("")
    # print("44444444444444444444444444444444444444444444")
    id_to_objects = {}
    for object_id, object_info in mask_dict.labels.items():
        start_pt = np.array([object_info.x1, object_info.y1], dtype=np.float32)
        end_pt = np.array([object_info.x2, object_info.y2], dtype=np.float32)
        bbox = np.array([start_pt, end_pt], dtype=np.float32)
        camera_predictor.add_new_prompt(frame_idx=0, obj_id=object_id, bbox=bbox)

        id_to_objects[object_id] = object_info.class_name
    # print("5555555555555555555555555555555555555555555")
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
    # print("Performing detection with prompt")
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

#####################
# Main function using exposed functions with full functionality
#####################
def main():
    DEBUG_MODE = False
    
    #####################
    # ROSBridge Setup
    #####################
    ros = setup_ros_bridge()
    
    #####################
    # Configurable Parameters
    #####################
    INPUT_IMAGE_TYPE = "CompressedImage"  # "CompressedImage" or "Image"
    # IMAGE_MSG_TYPE = "Image"
    
    # Load ros parameters
    if INPUT_IMAGE_TYPE == "CompressedImage":
        INPUT_IMAGE_TOPIC = get_ros_param(ros, '/tvss_nav/color_compressed_topic', '/camera/color/image_raw/compressed')
    elif INPUT_IMAGE_TYPE == "Image":
        INPUT_IMAGE_TOPIC = get_ros_param(ros, '/tvss_nav/color_topic', '/camera/color/image_raw')
        print("[Warning] Uncompressed images may lead to serious delays. Consider switching to CompressedImage format.")
    else:
        raise ValueError(f"Unsupported INPUT_IMAGE_TYPE: {INPUT_IMAGE_TYPE}")
    CAMERA_INFO_TOPIC = get_ros_param(ros, '/tvss_nav/color_info_topic', '/camera/color/camera_info')
    VISUAL_MASK_TOPIC = get_ros_param(ros, '/tvss_nav/visual_mask_topic', '/segmented_image/visual_mask/compressed')
    LABEL_MASK_TOPIC = get_ros_param(ros, '/tvss_nav/label_mask_topic', '/segmented_image/label_mask/compressed')
    SEGMENT_PROMPT_TOPIC = get_ros_param(ros, '/tvss_nav/segment_prompt_topic', '/segment_prompt')
    INSTANCE_CLASS_TOPIC = get_ros_param(ros, '/tvss_nav/inst_class_topic', '/instance_class_dict')
    ARENA_RESET_TOPIC = '/scenario_reset'    # published by arena task_manager

    HEIGHT = 480
    WIDTH = 640

    # Model and checkpoint settings
    SAM2_CHECKPOINT = "checkpoints/sam2.1_hiera_large.pt"
    MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
    MODEL_ID = "IDEA-Research/grounding-dino-base"

    cfg = OmegaConf.load("sam2/" + MODEL_CFG)
    use_trt = cfg.model.get("use_trt", None)
    
    if use_trt:
        model_name = os.path.splitext(os.path.basename(MODEL_CFG))[0]  # "sam2.1_hiera_large"
        trt_engine_name = model_name.replace("sam2.1_", "") + "_image_encoder.trt"
        os.environ["SAM2_TRT_ENGINE_PATH"] = os.path.join(os.path.dirname(__file__), "tensorrt", "trt", trt_engine_name)

    # parameters log in the terminal
    print("\n[Info] GSAM2 Configuration:")
    print(f"  - Input image type         : {INPUT_IMAGE_TYPE}")
    print(f"  - Input image topic        : {INPUT_IMAGE_TOPIC}")
    print(f"  - Camera info topic        : {CAMERA_INFO_TOPIC}")
    print(f"  - Visual mask topic        : {VISUAL_MASK_TOPIC}")
    print(f"  - Label mask topic         : {LABEL_MASK_TOPIC}")
    print(f"  - Segment prompt topic     : {SEGMENT_PROMPT_TOPIC}")
    print(f"  - Instance class topic     : {INSTANCE_CLASS_TOPIC}")
    print(f"  - Arena reset topic        : {ARENA_RESET_TOPIC}")
    print(f"  - Default image resolution : {WIDTH}x{HEIGHT}")
    print(f"  - SAM2 model config        : {MODEL_CFG}")
    print(f"  - SAM2 checkpoint          : {SAM2_CHECKPOINT}")
    print(f"  - GroundingDINO model ID   : {MODEL_ID}")
    print(f"  - TensorRT enabled         : {use_trt}")
    if use_trt:
        print(f"  - TensorRT engine name     : {trt_engine_name}")
    print()

    #####################
    # State variables (now local to main)
    #####################
    camera_info_received = False
    global_msg = None
    msg_lock = Lock()
    text_prompt = None
    restart_detection = False
    # text_prompt = "person"      # 固定检测类别（不要带句号）
    # restart_detection = True     # 启动后立刻做一次初始化检测
    global_reset_signal = False

    #####################
    # ROS Subscribers and Publishers
    #####################
    def image_msg_parser(msg):
        if INPUT_IMAGE_TYPE == "CompressedImage":
            frame = decode_compressed_image(msg)
        elif INPUT_IMAGE_TYPE == "Image":
            frame = decode_image(msg, HEIGHT, WIDTH)
        else:
            print("Unsupported IMAGE_MSG_TYPE:", INPUT_IMAGE_TYPE)
        frame_ts = [msg["header"]["stamp"]['secs'], msg["header"]["stamp"]['nsecs']]
        frame_link = msg["header"]["frame_id"]

        return frame, frame_ts, frame_link

    def image_callback(msg):
        nonlocal global_msg
        if not camera_info_received: 
            print("[Warning] Camera info not received yet. Waiting for camera info...")
            return
        with msg_lock:
            global_msg = msg
    
    def handle_camera_info(msg):
        nonlocal HEIGHT, WIDTH, camera_info_received
        if 'height' in msg and 'width' in msg:
            HEIGHT = msg['height']
            WIDTH = msg['width']
            if not camera_info_received:
                print(f"\n[INFO] Camera info updated according to ros topic \"{CAMERA_INFO_TOPIC}\": Height={HEIGHT}, Width={WIDTH}")
            camera_info_received = True
        else:
            print("[Warning] Camera info message does not contain height/width. Using default values.")
        
    def handle_text_prompt(msg):
        nonlocal text_prompt, restart_detection
        print(f"[GSAM2] 收到分割请求: {text_prompt}")
        # text_prompt = msg['data']  # Extract string from ROS message
        raw = msg['data']
        prompt = raw.strip()
        if not prompt.endswith('.'):
            prompt += '.'
        text_prompt = prompt
        restart_detection = True
        if DEBUG_MODE:
            print(f"\n[INFO] Received text prompt: {text_prompt}")

    def task_reset_signal(msg):
        nonlocal global_reset_signal
        global_reset_signal = True

    # subscribers 
    rgb_subscriber = create_subscriber(ros, INPUT_IMAGE_TOPIC, 'sensor_msgs/' + INPUT_IMAGE_TYPE, image_callback)
    camera_info_subscriber = create_subscriber(ros, CAMERA_INFO_TOPIC, 'sensor_msgs/CameraInfo', handle_camera_info)
    text_subscriber = create_subscriber(ros, SEGMENT_PROMPT_TOPIC, 'std_msgs/String', handle_text_prompt)
    reset_subscriber = create_subscriber(ros, ARENA_RESET_TOPIC, 'std_msgs/Int16', task_reset_signal)
    # publishers
    overlay_publisher = create_publisher(ros, VISUAL_MASK_TOPIC, 'sensor_msgs/CompressedImage')
    mask_publisher = create_publisher(ros, LABEL_MASK_TOPIC, 'sensor_msgs/CompressedImage')
    inst_class_publisher = create_publisher(ros, INSTANCE_CLASS_TOPIC, 'tvss_nav/StringStamped')
    
    if DEBUG_MODE:
        print(f"Created publisher for topic: {VISUAL_MASK_TOPIC}")
    
    #####################
    # Model initialization
    #####################
    torch.autocast(device_type=device, dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    camera_predictor = build_sam2_camera_predictor(MODEL_CFG, SAM2_CHECKPOINT)
    sam2_image_model = build_sam2(MODEL_CFG, SAM2_CHECKPOINT, device=device)
    image_predictor = SAM2ImagePredictor(sam2_image_model)
    # from transformers import PretrainedProcessor
    # # tell it to also look for preprocessor_config.json
    # PretrainedProcessor.config_struct = PretrainedProcessor.config_struct._replace(
    #     config_file_names=("processor_config.json", "preprocessor_config.json")
    # )
    # from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    # from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    # print("begin loading Grounded-SAM model and processor...")

    LOCAL_PATH = "/home/gavin0576/models/grounding-dino-base"

    processor = AutoProcessor.from_pretrained(LOCAL_PATH, local_files_only=True)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(LOCAL_PATH, local_files_only=True).to(device)

    
    # processor = AutoProcessor.from_pretrained(MODEL_ID,token=None)
    # grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID,token=None).to(device)
    # print("Finished loading Grounded-SAM model and processor.")
    id_to_objects = None
    sam2_masks = MaskDictionaryModel()

    rate = 2
    detection_timeout = 5
    # detection_timeout = np.inf

    last_detect_time = time.time()

    initialized = False

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
                # print("start-------------------------------------------------------------")
                try:
                    # print("trytrytrytrytrytry")
                    if not initialized:
                        # if DEBUG_MODE:
                        print(f"[INFO] Initializing detection with prompt: {text_prompt}")

                        id_to_objects, mask_dict = perform_init_detection(
                            processor, grounding_model, camera_predictor, image_predictor, locked_frame, text_prompt, WIDTH, HEIGHT)
                        sam2_masks = mask_dict

                        initialized = True
                        print(f"[INFO] Initialized with {len(id_to_objects)} objects.")
                    else:
                        try:
                            out_obj_ids, out_mask_logits, frame_resized = track_frame(camera_predictor, locked_frame, WIDTH, HEIGHT)
                            for obj_id, mask_logit in zip(out_obj_ids, out_mask_logits):
                                mask_binary = (mask_logit > 0.0).squeeze(0)
                                if obj_id in sam2_masks.labels and mask_binary.sum() > 0:
                                    obj_info = sam2_masks.labels[obj_id]
                                    obj_info.mask = mask_binary
                                    obj_info.update_box()
                            print(f"[INFO] Tracking {len(out_obj_ids)} objects.")
                        except Exception as e:
                            print(f"[Warning] All tracking lost.")
                            # print(f"\n[Error] {e}")
                            time.sleep(rate)
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
                    print("omgomgomgomgomgomgomgomgomgomgomgomgomgomgomgomgomgomgomgomg")
                    time.sleep(rate)
                    continue
                except Exception as e:
                    print(f"[Error] Detection failed: {e}")
                    time.sleep(rate)
                    continue
            if text_prompt is not None:
                print(f"[INFO] Processing frame at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(locked_frame_ts[0]))} with prompt1: {text_prompt}")
            if not initialized:
                print("[Warning] Predictor not initialized. Waiting for initialization...")
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
            # print("start visualizing detections...")
            visual_mask = visualize_detections(frame_resized, out_obj_ids, out_mask_logits, id_to_objects)
            # cv2.imshow("Segmented Frame", visual_mask)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     print("Exiting main")
            #     break
            # print("start visualizing masks...")
            publish_visual_mask(overlay_publisher, visual_mask, locked_frame_ts, locked_frame_link)
            publish_instance_class_dict(inst_class_publisher, id_to_objects, locked_frame_ts, locked_frame_link)
            print("start publishing label mask...xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
            publish_label_mask(mask_publisher, frame_resized, out_obj_ids, out_mask_logits, locked_frame_ts, locked_frame_link)
            
            time.sleep(rate)
    except KeyboardInterrupt:
        print("Program interrupted")
    finally:
        # Cleanup
        cv2.destroyAllWindows()
        ros.terminate()

def publish_visual_mask(publisher, overlay, timestamp, frame_link):
    '''
    Publish the overlay image to a ROS topic.
    Args:
        publisher: roslibpy publisher for CompressedImage.
        overlay: The overlay image to be published.
        timestamp: (secs, nsecs) tuple or None.
        frame_link (str): Frame ID for Header.
    '''
    compressed_msg = create_compressed_image_message(overlay, format='jpg', quality=80, timestamp=timestamp, frame_link=frame_link)
    publisher.publish(roslibpy.Message(compressed_msg))

def publish_instance_class_dict(publisher, id_to_objects, timestamp, frame_link):
    """
    Publish instance-class mapping as a custom message via roslibpy with dict-based formatting.

    Args:
        publisher: roslibpy publisher for InstanceClassDict.
        id_to_objects: Dictionary mapping object IDs to class labels.
        timestamp: (secs, nsecs) tuple or None.
        frame_link (str): Frame ID for Header.
    """
    # Handle timestamp
    if timestamp is None:
        now = time.time()
        secs = int(now)
        nsecs = int((now - secs) * 1e9)
    else:
        secs, nsecs = timestamp
        
    # Construct final ROS message dict
    class_dict_msg = {
        'header': {
            'stamp': {'secs': secs, 'nsecs': nsecs},
            'frame_id': frame_link
        },
        'data': json.dumps(id_to_objects)
    }

    # Publish as roslibpy.Message
    publisher.publish(roslibpy.Message(class_dict_msg))
    
def publish_label_mask(publisher, frame_resized, out_obj_ids, out_mask_logits, timestamp, frame_link):
    """ 
    Publish a Mask image with instance IDs.

    - The "mask" values are no longer 0/255, but instead instance_id itself.
    - Uses PNG compression to reduce bandwidth.
    - Adds erosion to clean mask edges.
    
    Args:
        publisher: roslibpy publisher for CompressedImage.
        frame_resized: The resized frame used for tracking.
        out_obj_ids: List of object IDs.
        out_mask_logits: Segmentation mask logits.
        timestamp: (secs, nsecs) tuple or None.
        frame_link (str): Frame ID for Header.
    """

    all_mask = np.zeros((frame_resized.shape[0], frame_resized.shape[1]), dtype=np.uint8)  # HxW
    print(f"[INFO] Publishing label mask with {len(out_obj_ids)} objects.")
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

    publisher.publish(roslibpy.Message(mask_msg))
    print(f"[INFO] Label mask published with {len(out_obj_ids)} objects.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Exception occurred:", e)
    finally:
        cv2.destroyAllWindows()
