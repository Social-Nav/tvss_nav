#!/usr/bin/env python3
import os,sys
# THIS_FILE = os.path.abspath(__file__)
# PKG_ROOT  = os.path.dirname(os.path.dirname(THIS_FILE))
THIS_FILE = os.path.abspath(__file__)
PKG_ROOT  = os.path.dirname(THIS_FILE)
sys.path.insert(0, PKG_ROOT)
import threading
import torch
import base64
import rospy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped
from PIL import Image
from io import BytesIO

import cv2
import numpy as np

from robopoint.model.builder import load_pretrained_model
from robopoint.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from robopoint.conversation import conv_templates, SeparatorStyle
from robopoint.utils import disable_torch_init
from robopoint.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

class RosLocalInfer:
    def __init__(self):
        self.image_topic  = rospy.get_param('~image_topic', '/camera/color/image_raw/compressed')
        self.prompt_topic = rospy.get_param('~prompt_topic', '/user_query')
        self.output_topic = rospy.get_param('~output_topic', '/pixel_subgoal')

        self.latest_image_b64 = None
        self.latest_prompt    = None
        self.latest_header    = None
        self.lock = threading.Lock()

        model_path = rospy.get_param('~model_path',
            '~/arena_ws/src/tvss_nav/scripts/tvss_nav/sampler/robopoint_sampler/robopoint/wentao-yuan/robopoint-v1-vicuna-v1.5-7b-lora')
        model_base = os.path.expanduser(
            rospy.get_param('~model_base',
                            '~/arena_ws/src/tvss_nav/scripts/tvss_nav/sampler/robopoint_sampler/robopoint/wentao-yuan/vicuna-7b-v1.5'))
        device     = 'cuda' if torch.cuda.is_available() else 'cpu'
        # rospy.loginfo(f"[RosLocalInfer] Loading model {model_path} (base {model_base}) on {device}…")
        # self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
        #     model_path=os.path.expanduser(model_path),
        #     model_base=model_base,
        #     model_name=None,
        #     load_8bit=False,
        #     load_4bit=False,
        #     device=device
        expanded = os.path.expanduser(model_path)
        model_name = rospy.get_param('~model_name', os.path.basename(expanded))
        rospy.loginfo(f"[RosLocalInfer] Loading model {expanded} (base {model_base}) "
                      f"with name {model_name} on {device}…")
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            model_path=expanded,
            model_base=model_base,
            model_name=model_name,
            load_8bit=False,
            load_4bit=False,
            device=device
        )
        self.device = device

        rospy.Subscriber(self.image_topic, CompressedImage, self.image_cb,  queue_size=1)
        rospy.Subscriber(self.prompt_topic, String,        self.prompt_cb, queue_size=1)
        # self.pub = rospy.Publisher(self.output_topic, String, queue_size=1)
        self.stdin_pub = rospy.Publisher(self.prompt_topic, String, queue_size=1)
        self.pub = rospy.Publisher(self.output_topic, PointStamped, queue_size=1)

        threading.Thread(target=self._stdin_prompt_loop, daemon=True).start()

        rospy.loginfo("[RosLocalInfer] Ready — waiting for image & prompt.")

    def _stdin_prompt_loop(self):
        rospy.loginfo("[RosLocalInfer] —— input query, and press enter ——")
        while not rospy.is_shutdown():
            try:
                q = input(">> user_query: ").strip()
                if q:
                    rospy.loginfo(f"[RosLocalInfer] <<< Manual send >>> “{q}”")
                    self.stdin_pub.publish(String(data=q))
            except EOFError:
                break
            except Exception as e:
                rospy.logwarn(f"[RosLocalInfer] stdin loop error: {e}")

    def image_cb(self, msg: CompressedImage):
        with self.lock:
            if isinstance(msg.data, bytes):
                self.latest_image_b64 = base64.b64encode(msg.data).decode('utf-8')
            else:
                self.latest_image_b64 = msg.data
            self.latest_header = msg.header
        self.try_infer()

    def prompt_cb(self, msg: String):
        rospy.loginfo(f"[RosLocalInfer] <<< Received user_query >>> “{msg.data}”")
        with self.lock:
            self.latest_prompt = msg.data.strip()
        self.try_infer()

    def try_infer(self):
        with self.lock:
            if not self.latest_image_b64 or not self.latest_prompt:
                return
            img_b64 = self.latest_image_b64
            prompt  = self.latest_prompt
            self.latest_image_b64 = None
            self.latest_prompt    = None

        threading.Thread(target=self.run_infer, args=(img_b64, prompt), daemon=True).start()

    def run_infer(self, img_b64: str, prompt: str):
        if not img_b64:
            rospy.logwarn("[RosLocalInfer] There is no image right now. Skip inference.")
            return

        rospy.loginfo(f"[RosLocalInfer] Inference for prompt: “{prompt}”")

        img_data = base64.b64decode(img_b64)
        img = Image.open(BytesIO(img_data)).convert("RGB")

        img_tensor = process_images([img], self.image_processor, self.model.config)[0] \
                        .to(self.device).half().unsqueeze(0)
        print(f"[DEBUG] image mode={img.mode}, size={img.size}")           
        print(f"[DEBUG] img_tensor.shape={tuple(img_tensor.shape)}")      
        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        cv2.imshow("DEBUG - current image", cv_img)
        cv2.waitKey(5000)  
        cv2.destroyWindow("DEBUG - current image")

        conv = conv_templates['llava_v1'].copy()
        if getattr(self.model.config, 'mm_use_im_start_end', False):
            img_block = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        else:
            img_block = DEFAULT_IMAGE_TOKEN

        instruction = (
            "Your answer should be formatted as a list of tuples, "
            "i.e. [(x1, y1), (x2, y2), ...], where each tuple contains "
            "the x and y coordinates of a point satisfying the conditions above. "
            "The coordinates should be between 0 and 1, indicating the normalized "
            "pixel locations of the points in the image.\n\n"
        )
        user_msg = img_block + "\n" + instruction + prompt
        conv.append_message(conv.roles[0], user_msg)
        conv.append_message(conv.roles[1], None)

        full_prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            full_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,           
                img_tensor,         
                [img.size],          
                do_sample=True,
                temperature=0.2,
                top_p=0.95,
                max_new_tokens=256,
                use_cache=True
            )

        result = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        print(f"\n>> user_query: {prompt}")
        print(f"\n>> system instruction: {instruction}")
        print(f"<< model_output_ids: {output_ids}")
        print(f"<< model_output: {result}")
        try:
            coords_norm = eval(result)  
        except Exception as e:
            rospy.logwarn(f"[RosLocalInfer] fail to read coordinates: {e}")
            coords_norm = []

        if coords_norm:
            img_np = np.array(img)                   
            vis = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            w, h = img.size                           
            pixel_pts = [(int(x * w), int(y * h)) for x, y in coords_norm]

            for (px, py) in pixel_pts:
                cv2.circle(vis, (px, py), radius=5, color=(0,255,0), thickness=-1)

            avg_x = int(sum(px for px, _ in pixel_pts) / len(pixel_pts))
            avg_y = int(sum(py for _, py in pixel_pts) / len(pixel_pts))
            cv2.circle(vis, (avg_x, avg_y), radius=8, color=(0,0,255), thickness=-1)

            cv2.imshow("RoboPoint Visualization", vis)
            cv2.waitKey(5000)
            cv2.destroyWindow("RoboPoint Visualization")

            rospy.loginfo(f"[RosLocalInfer] Average subgoal pixel: ({avg_x}, {avg_y})")
        else:
            rospy.loginfo("[RosLocalInfer] No valid points to visualize.")
        rospy.loginfo(f"[RosLocalInfer] Result → {result}")
        # self.pub.publish(String(data=result))
        ps = PointStamped()
        ps.header.stamp = self.latest_header.stamp             
        ps.header.frame_id = self.latest_header.frame_id      
        ps.point.x = avg_x
        ps.point.y = avg_y
        ps.point.z = 0.0
        self.pub.publish(ps)

        sys.stdout.write("\n>> user_query: ")
        sys.stdout.flush()


if __name__ == "__main__":
    rospy.init_node('ros_local_infer', anonymous=False)
    RosLocalInfer()
    rospy.spin()
