#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VLM-Social-Nav (article-aligned, single file, threaded)
- Implements Algorithm 1 from VLM-Social-Nav (Csocial scoring + action sampling)
- Perception-gated VLM queries; high-rate cmd publisher; low-rate planner
- Prompt enforces:  "Move DIRECTION with SPEED" (left|straight|right × slow down|constant|speed up|stop)

Added (latency metrics):
- Measure end-to-end latency for each VLM call (send -> receive), log per-call
- Sliding stats over recent calls (avg/min/max), ROS topic /vlm_latency (std_msgs/Float32)
- Final summary (total calls & global average) on shutdown

Connections (via rosbridge):
  /camera/color/image_raw/compressed  (sensor_msgs/CompressedImage)
  /camera/depth/image_raw             (sensor_msgs/Image)        [optional]
  /tf, /tf_static                     (tf2_msgs/TFMessage)
  /cmd_vel                            (geometry_msgs/Twist)

Config: put a YAML at ./config/params.yaml or set VLM_SOCIAL_CFG env var
  alpha, beta, gamma: weights for C_goal, C_obst, C_social
  wl, wa: weights inside C_social
  publish_frequency, planner_frequency
  candidate_actions: [[v,w], ...]
  vlm_every_n: call VLM every N planner ticks (>=1)
  use_vlm_only_when_social: bool; gate VLM on social cue
  yolo_conf: float
  v_max, w_max: saturation
  cmd_vel_topic: string
  vlm_api_key: string or use env OPENAI_API_KEY
"""

import os
import time
import base64
import threading
from collections import deque

import numpy as np
import cv2
import yaml

# ----------------- OpenAI SDK shim -----------------
USE_OPENAI_V1 = False
try:
    from openai import OpenAI  # >=1.x
    USE_OPENAI_V1 = True
except Exception:
    import openai               # legacy fallback

# ----------------- roslibpy -----------------
try:
    from roslibpy import Ros, Topic
except ImportError:
    print("Error: roslibpy not found. pip install roslibpy")
    raise

# ----------------- YOLO (optional) -----------------
YOLO_BACKEND = "none"
try:
    from ultralytics import YOLO   # YOLOv8/11
    YOLO_BACKEND = "ultralytics"
except Exception:
    try:
        import torch
        YOLO_BACKEND = "yolov5"
    except Exception:
        YOLO_BACKEND = "none"


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def _mat_to_quat(R):
    m00, m01, m02 = R[0,0], R[0,1], R[0,2]
    m10, m11, m12 = R[1,0], R[1,1], R[1,2]
    m20, m21, m22 = R[2,0], R[2,1], R[2,2]
    tr = m00 + m11 + m22
    if tr > 0:
        S = (tr + 1.0) ** 0.5 * 2.0
        w = 0.25 * S
        x = (m21 - m12) / S
        y = (m02 - m20) / S
        z = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = ((1.0 + m00 - m11 - m22) ** 0.5) * 2.0
        w = (m21 - m12) / S
        x = 0.25 * S
        y = (m01 + m10) / S
        z = (m02 + m20) / S
    elif m11 > m22:
        S = ((1.0 + m11 - m00 - m22) ** 0.5) * 2.0
        w = (m02 - m20) / S
        x = (m01 + m10) / S
        y = 0.25 * S
        z = (m12 + m21) / S
    else:
        S = ((1.0 + m22 - m00 - m11) ** 0.5) * 2.0
        w = (m10 - m01) / S
        x = (m02 + m20) / S
        y = (m12 + m21) / S
        z = 0.25 * S
    return x, y, z, w


def _make_T(trans_dict, rot_dict):
    tx, ty, tz = float(trans_dict.get('x', 0.0)), float(trans_dict.get('y', 0.0)), float(trans_dict.get('z', 0.0))
    x, y, z, w = float(rot_dict.get('x', 0.0)), float(rot_dict.get('y', 0.0)), float(rot_dict.get('z', 0.0)), float(rot_dict.get('w', 1.0))
    n = (x*x + y*y + z*z + w*w) ** 0.5
    if n == 0.0:
        x=y=z=0.0; w=1.0
    else:
        x/=n; y/=n; z/=n; w/=n
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w),   2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z),   2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w),   1-2*(x*x+y*y)]
    ], dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3,:3] = R
    T[:3, 3] = [tx, ty, tz]
    return T


def _invert_T(T):
    R = T[:3,:3]
    t = T[:3, 3]
    Tinv = np.eye(4, dtype=np.float64)
    Tinv[:3,:3] = R.T
    Tinv[:3, 3] = -R.T @ t
    return Tinv


# ----------------- Perception -----------------
class SocialPerception:
    """
    Detects social cues (person, door). Returns flags and rough geometry to help prompt.
    For simplicity:
      - "doctor" is proxied by class "person"
      - offset in [-1, 1] (left negative, right positive)
      - scale  ≈ bbox_area / image_area ∈ [0, 1]
    """
    def __init__(self, conf=0.35):
        self.conf = float(conf)
        self.mode = YOLO_BACKEND
        self.model = None
        self.names = {}
        self.labels_of_interest = {"person", "door"}
        try:
            if self.mode == "ultralytics":
                self.model = YOLO("yolov8n.pt")
                print("[Perception] ultralytics YOLO loaded.")
            elif self.mode == "yolov5":
                import torch
                self.model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
                self.model.conf = self.conf
                self.names = self.model.names
                print("[Perception] yolov5 loaded.")
            else:
                print("[Perception] YOLO disabled.")
        except Exception as e:
            print(f"[Perception] init failed: {e}")
            self.mode = "none"
            self.model = None

    def detect(self, bgr_image):
        if bgr_image is None or self.model is None or self.mode == "none":
            return []
        try:
            labels = []
            if self.mode == "ultralytics":
                res = self.model.predict(bgr_image, imgsz=640, conf=self.conf, verbose=False)
                for r in res:
                    if r.boxes is None:
                        continue
                    for c in r.boxes.cls.cpu().numpy().astype(int):
                        name = r.names.get(c, "")
                        if name in self.labels_of_interest or name == "person":
                            labels.append(name or "person")
            elif self.mode == "yolov5":
                results = self.model(bgr_image, size=640)
                det = results.xyxy[0].cpu().numpy()
                for *box, conf, cls_id in det:
                    name = self.names.get(int(cls_id), "")
                    if name in self.labels_of_interest or name == "person":
                        labels.append(name or "person")
            return labels
        except Exception as e:
            print(f"[Perception] inference error: {e}")
            return []

    def get_person_geo(self, bgr_image):
        if bgr_image is None or self.model is None or self.mode == "none":
            return False, 0.0, 0.0
        H, W = bgr_image.shape[:2]
        try:
            boxes = []
            if self.mode == "ultralytics":
                res = self.model.predict(bgr_image, imgsz=640, conf=self.conf, verbose=False)
                for r in res:
                    if r.boxes is None:
                        continue
                    xyxy = r.boxes.xyxy.cpu().numpy()
                    cls  = r.boxes.cls.cpu().numpy().astype(int)
                    conf = r.boxes.conf.cpu().numpy()
                    for (x1,y1,x2,y2), c, p in zip(xyxy, cls, conf):
                        name = r.names.get(int(c), "")
                        if name == "person":
                            boxes.append((float(p), float(x1), float(y1), float(x2), float(y2)))
            elif self.mode == "yolov5":
                results = self.model(bgr_image, size=640)
                det = results.xyxy[0].cpu().numpy()
                for x1,y1,x2,y2,conf,cls_id in det:
                    name = self.names.get(int(cls_id), "")
                    if name == "person":
                        boxes.append((float(conf), float(x1), float(y1), float(x2), float(y2)))

            if not boxes:
                return False, 0.0, 0.0
            boxes.sort(key=lambda t: t[0], reverse=True)
            _, x1, y1, x2, y2 = boxes[0]
            cx = (x1 + x2) * 0.5
            area = max(1.0, (x2 - x1) * (y2 - y1))
            img_area = float(W * H)
            offset = (cx - W*0.5) / (W*0.5)
            scale  = clamp(area / img_area, 0.0, 1.0)
            return True, float(offset), float(scale)
        except Exception as e:
            print(f"[Perception] get_person_geo error: {e}")
            return False, 0.0, 0.0


# ----------------- VLM cost scorer -----------------
class VLMCostScorer:
    def __init__(self, api_key, wl=1.0, wa=1.0):
        self.wl = float(wl)
        self.wa = float(wa)
        # discrete policy anchor values
        self.base_v = 0.45
        self.dir_to_w = {"left": +0.15, "straight": 0.0, "right": -0.15}
        self.spd_to_v = {
            "slow down": 0.15,
            "constant":  0.45,
            "speed up":  0.50,
            "stop":      0.00,
        }
        # Prompt 
        self.prompt_template = (
            "Task: How will you navigate with respect to the person(s) in view?\n"
            "Follow general walking etiquette.\n\n"
            "Ego state:\n"
            "- heading: {heading}\n"
            "- linear velocity: {speed:.2f}\n"
            "- person_offset: {person_offset:.2f}  # [-1..1], left negative, right positive\n"
            "- person_scale:  {person_scale:.3f}   # bbox area ratio in view\n\n"
            "Remember:\n"
            "- Keep to the right when passing (adjust if your locale keeps left).\n"
            "- Do not obstruct others’ paths; yield when appropriate.\n"
            "- Maintain safe distance; slow/stop if too close.\n\n"
            "Answer Format:\n"
            "Move DIRECTION with SPEED\n"
            "- DIRECTION in [left, straight, right]\n"
            "- SPEED in [slow down, constant, speed up, stop]"
        )
        self.use_v1 = USE_OPENAI_V1
        if self.use_v1:
            self.client = OpenAI(api_key=api_key)
        else:
            openai.api_key = api_key

    def _build_messages(self, heading, speed, image_bgr=None, person_offset=0.0, person_scale=0.0):
        content = [{
            "type": "text",
            "text": self.prompt_template.format(
                heading=heading, speed=speed,
                person_offset=person_offset, person_scale=person_scale,
            )
        }]
        if image_bgr is not None:
            ok, buf = cv2.imencode(".jpg", image_bgr)
            if ok:
                b64 = base64.b64encode(buf).decode("utf-8")
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        return [{"role": "user", "content": content}]

    def query(self, heading="straight", speed=0.25, image_bgr=None, person_offset=0.0, person_scale=0.0):
        """Return (text_output, latency_sec). Latency = send->receive; includes network+server+SDK."""
        messages = self._build_messages(heading, speed, image_bgr, person_offset, person_scale)
        t_send = time.time()
        try:
            if self.use_v1:
                resp = self.client.chat.completions.create(
                    model=("gpt-4o" if image_bgr is not None else "gpt-4o-mini"),
                    messages=messages,
                    max_tokens=48,
                    temperature=0.3,
                )
                out = resp.choices[0].message.content.strip()
            else:
                resp = openai.ChatCompletion.create(
                    model=("gpt-4-vision-preview" if image_bgr is not None else "gpt-4o-mini"),
                    messages=messages,
                    max_tokens=48,
                    temperature=0.3,
                )
                out = resp["choices"][0]["message"]["content"].strip()
            t_recv = time.time()
            latency = t_recv - t_send
            print(f"[VLM Output] {out}  |  latency={latency*1000:.1f} ms")
            return out, latency
        except Exception as e:
            t_recv = time.time()
            latency = t_recv - t_send
            print(f"[VLM ERROR] {e}  |  latency={latency*1000:.1f} ms")
            return "Move straight with constant", latency

    def parse_action(self, text):
        t = text.lower()
        if "left" in t:
            direction = "left"
        elif "right" in t:
            direction = "right"
        elif any(k in t for k in ["straight", "forward", "ahead"]):
            direction = "straight"
        else:
            direction = "straight"

        if "stop" in t:
            speed_w = "stop"
        elif "slow" in t or "decel" in t:
            speed_w = "slow down"
        elif "speed up" in t or "faster" in t or "accel" in t:
            speed_w = "speed up"
        elif "constant" in t or "maintain" in t or "steady" in t:
            speed_w = "constant"
        else:
            speed_w = "constant"

        v = self.spd_to_v.get(speed_w, self.base_v)
        w = self.dir_to_w.get(direction, 0.0)
        return v, w

    def c_social(self, action_vw, suggested_vw):
        v, w = action_vw
        v_h, w_h = suggested_vw
        return self.wl * abs(v - v_h) + self.wa * abs(w - w_h)


# ----------------- Main Node -----------------
class VLMSocialNavNode:
    def __init__(self):
        # ROS
        host = os.environ.get("ROSLIBPY_HOST", "localhost")
        port = int(os.environ.get("ROSLIBPY_PORT", "9090"))
        self.ros = Ros(host, port)
        self.ros.run()
        while not self.ros.is_connected:
            time.sleep(0.02)
        

        # Config
        rel_cfg = os.environ.get("VLM_SOCIAL_CFG", os.path.join("config", "params.yaml"))
        base_dir = os.path.dirname(os.path.realpath(__file__))
        cfg_path = os.path.join(base_dir, rel_cfg) if not os.path.isabs(rel_cfg) else rel_cfg
        self.cfg = self._load_yaml(cfg_path)

        # Topics
        self.rgb_topic = self.cfg.get("rgb_topic", "/camera/color/image_raw/compressed")
        self.depth_topic = self.cfg.get("depth_topic", "/camera/depth/image_raw")
        self.cmd_vel_topic = self.cfg.get("cmd_vel_topic", "/jackal/cmd_vel")
        self.cmd_pub = Topic(self.ros, self.cmd_vel_topic, 'geometry_msgs/Twist')

        # Subscribers
        Topic(self.ros, self.rgb_topic, 'sensor_msgs/CompressedImage').subscribe(self._rgb_cb)
        Topic(self.ros, self.depth_topic, 'sensor_msgs/Image').subscribe(self._depth_cb)
        Topic(self.ros, '/tf', 'tf2_msgs/TFMessage').subscribe(lambda m: self._tf_cb(m, False))
        Topic(self.ros, '/tf_static', 'tf2_msgs/TFMessage').subscribe(lambda m: self._tf_cb(m, True))

        # Buffers
        self.width, self.height = 640, 480
        self.rgb = None
        self.depth = None
        self.camera_frame = None
        self.base_frame = self.cfg.get("base_frame", "base_link")
        self.odom_frame = self.cfg.get("odom_frame", "odom")

        # Weights & limits
        self.alpha = float(self.cfg.get("alpha", 1.0))
        self.beta  = float(self.cfg.get("beta", 1.0))
        self.gamma = float(self.cfg.get("gamma", 1.0))
        self.wl    = float(self.cfg.get("wl", 1.0))
        self.wa    = float(self.cfg.get("wa", 1.0))
        self.v_max = float(self.cfg.get("v_max", 0.8))
        self.w_max = float(self.cfg.get("w_max", 1.2))

        # Action set & rates
        self.candidate_actions = self.cfg.get("candidate_actions", [
            [0.20, +0.60], [0.40, +0.30], [0.42, 0.00], [0.40, -0.30], [0.20, -0.60], [0.00, 0.00]
        ])
        self.publish_frequency = float(self.cfg.get("publish_frequency", 20.0))
        self.planner_frequency = float(self.cfg.get("planner_frequency", 10.0))
        self.action_duration_sec = float(self.cfg.get("action_duration_sec", 0.8))

        # VLM gating
        self.vlm_every_n = int(self.cfg.get("vlm_every_n", 5))
        self._vlm_tick = 0
        self._last_vlm_vw = (0.30, 0.0)
        self._use_gate = bool(self.cfg.get("use_vlm_only_when_social", True))

        # Locks / command state
        self._cmd_lock = threading.Lock()
        self._current_cmd = (0.0, 0.0)
        self._held_act = (0.0, 0.0)
        self._hold_until = 0.0

        # Modules
        self.perception = SocialPerception(conf=self.cfg.get("yolo_conf", 0.35))
        self.vlm = VLMCostScorer(
            api_key=self.cfg.get("vlm_api_key", os.environ.get("OPENAI_API_KEY", "")),
            wl=self.wl, wa=self.wa
        )
        if not (self.cfg.get("vlm_api_key") or os.environ.get("OPENAI_API_KEY")):
            print("[WARN] OPENAI_API_KEY is empty; VLM calls will fail over to default suggestion.")

        # TF graphs
        self.tf_graph_dynamic = {}
        self.tf_graph_static = {}
        threading.Thread(target=self._cleaner_loop, daemon=True).start()

        # ---- Latency metrics (new) ----
        self._last_vlm_latency = 0.0
        self._lat_hist = deque(maxlen=100)                 # recent sliding window
        self.lat_count = 0                                 # total calls
        self.lat_sum = 0.0                                 # total latency sum
        self.lat_pub = Topic(self.ros, '/vlm_latency', 'std_msgs/Float32')  # for rqt_plot

        # Threads
        threading.Thread(target=self._planner_loop, daemon=True).start()
        threading.Thread(target=self._publisher_loop, daemon=True).start()
        print(f"[Node] VLM-Social-Nav ready. Publishing {self.cmd_vel_topic}")

    # ------------- YAML -------------
    def _load_yaml(self, path):
        try:
            with open(path, "r") as f:
                data = yaml.safe_load(f) or {}
            print(f"[CFG] Loaded: {path}")
            return data
        except Exception as e:
            print(f"[CFG] Failed to load {path}: {e}. Using defaults.")
            return {
                "alpha": 1.0, "beta": 1.0, "gamma": 1.0,
                "publish_frequency": 20.0,
                "planner_frequency": 5.0,
                "wl": 1.0, "wa": 1.0,
                "candidate_actions": [[0.30,0.0],[0.30,0.3],[0.30,-0.3],[0.10,0.0]],
                "vlm_api_key": os.environ.get("OPENAI_API_KEY",""),
                "cmd_vel_topic": "/jackal/cmd_vel"
            }

    # ------------- Housekeeping -------------
    def _cleaner_loop(self):
        while True:
            try:
                self._cleanup_old_transforms(5.0)
            except Exception:
                pass
            time.sleep(1.0)

    # ------------- Callbacks -------------
    def _rgb_cb(self, msg):
        try:
            data = msg.get('data', None)
            if data is None:
                return
            np_arr = np.frombuffer(base64.b64decode(data), dtype=np.uint8)
            rgb = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            self.rgb = rgb
            if self.camera_frame is None:
                fid = msg.get('header', {}).get('frame_id', '')
                if fid:
                    self.camera_frame = fid.lstrip('/')
        except Exception as e:
            print(f"[RGB] Bad message: {e}")

    def _depth_cb(self, msg):
        try:
            encoding = msg.get('encoding', '32FC1')
            raw = msg.get('data', b'')
            if isinstance(raw, str):
                try:
                    raw = base64.b64decode(raw)
                except Exception:
                    raw = raw.encode('latin-1', errors='ignore')
            H = int(msg.get('height', self.height))
            W = int(msg.get('width', self.width))
            if '32FC1' in encoding or '32FC' in encoding:
                np_arr = np.frombuffer(raw, dtype=np.float32)
                if np_arr.size == H*W:
                    self.depth = np.nan_to_num(np_arr.reshape((H, W)))
            elif '16UC1' in encoding:
                np_arr = np.frombuffer(raw, dtype=np.uint16)
                if np_arr.size == H*W:
                    self.depth = np.nan_to_num(np_arr.reshape((H, W)).astype(np.float32)/1000.0)
            else:
                np_arr = np.frombuffer(raw, dtype=np.uint8)
                if np_arr.size == H*W:
                    self.depth = np_arr.reshape((H, W)).astype(np.float32)
                else:
                    self.depth = np.zeros((H, W), dtype=np.float32)
        except Exception as e:
            print(f"[DEPTH] Bad message: {e}")

    def _store_edge(self, graph, parent, child, T, timestamp):
        graph.setdefault(parent, {})[child] = {'T': T, 'timestamp': timestamp}
        graph.setdefault(child, {})[parent] = {'T': _invert_T(T), 'timestamp': timestamp}

    def _tf_cb(self, msg, is_static=False):
        try:
            transforms = msg.get('transforms', [])
            if not transforms:
                return
            graph = self.tf_graph_static if is_static else self.tf_graph_dynamic
            for tfs in transforms:
                frame_id = tfs.get('header', {}).get('frame_id', '').lstrip('/')
                child_id = tfs.get('child_frame_id', '').lstrip('/')
                if not frame_id or not child_id:
                    continue
                tr = tfs.get('transform', {}).get('translation', {})
                rq = tfs.get('transform', {}).get('rotation', {})
                T = _make_T(tr, rq)
                self._store_edge(graph, frame_id, child_id, T, time.time())
        except Exception as e:
            print(f"[TF] parse error: {e}")

    def _merged_graph(self):
        merged = {}
        for a, nbrs in self.tf_graph_static.items():
            merged[a] = {b: info.copy() for b, info in nbrs.items()}
        for a, nbrs in self.tf_graph_dynamic.items():
            merged.setdefault(a, {})
            for b, info in nbrs.items():
                merged[a][b] = info
        return merged

    def _find_T_path(self, target, source):
        if target == source:
            return np.eye(4, dtype=np.float64)
        graph = self._merged_graph()
        if target not in graph:
            return None
        q = deque([target])
        acc_T = {target: np.eye(4, dtype=np.float64)}
        visited = set()
        while q:
            cur = q.popleft()
            if cur == source:
                return acc_T[cur]
            visited.add(cur)
            for nxt, info in graph.get(cur, {}).items():
                if nxt in visited:
                    continue
                T_acc = acc_T[cur] @ info['T']
                if nxt not in acc_T:
                    acc_T[nxt] = T_acc
                    q.append(nxt)
        return None

    def lookup_pose(self, target_frame, source_frame):
        try:
            target = str(target_frame).lstrip('/')
            source = str(source_frame).lstrip('/')
            T = self._find_T_path(target, source)
            if T is None:
                return np.zeros(3, dtype=np.float32), np.array([0,0,0,1], dtype=np.float32)
            R = T[:3,:3]; t = T[:3,3]
            qx, qy, qz, qw = _mat_to_quat(R)
            return t.astype(np.float32), np.array([qx, qy, qz, qw], dtype=np.float32)
        except Exception:
            return np.zeros(3, dtype=np.float32), np.array([0,0,0,1], dtype=np.float32)

    def _cleanup_old_transforms(self, max_age=5.0):
        now_t = time.time()
        to_del = []
        for a, nbrs in list(self.tf_graph_dynamic.items()):
            for b, info in list(nbrs.items()):
                if now_t - info.get('timestamp', now_t) > max_age:
                    to_del.append((a,b))
        for a,b in to_del:
            try:
                del self.tf_graph_dynamic[a][b]
                if not self.tf_graph_dynamic[a]:
                    del self.tf_graph_dynamic[a]
            except KeyError:
                pass

    # ------------- Threads -------------
    def _planner_loop(self):
        dt_plan = 1.0 / max(1e-3, self.planner_frequency)
        while True:
            t0 = time.time()
            if self.rgb is None:
                time.sleep(dt_plan); continue

            # 1) 感知
            entities = self.perception.detect(self.rgb)
            found_person, person_offset, person_scale = self.perception.get_person_geo(self.rgb)
            has_social = bool(entities) or found_person
            if found_person:
                print(f"[Perception] person: offset={person_offset:+.2f}, scale={person_scale:.3f}")

            # ---- 决定是否调用 VLM，并只调用一次 ----
            need_vlm = (not self._use_gate) or has_social
            call_vlm = (self._vlm_tick % max(1, self.vlm_every_n) == 0)

            if need_vlm and call_vlm:
                out, latency = self.vlm.query(
                    heading="straight",
                    speed=0.30,
                    image_bgr=self.rgb,
                    person_offset=(person_offset if found_person else 0.0),
                    person_scale=(person_scale if found_person else 0.0),
                )
                # --- latency metrics update ---
                self._last_vlm_latency = float(latency)
                self._lat_hist.append(self._last_vlm_latency)
                self.lat_count += 1
                self.lat_sum += self._last_vlm_latency
                # publish to ROS
                try:
                    self.lat_pub.publish({'data': self._last_vlm_latency})
                except Exception:
                    pass
                # sliding stats for logs
                lat_ms = self._last_vlm_latency * 1000.0
                avg_ms = (sum(self._lat_hist)/len(self._lat_hist))*1000.0
                min_ms = (min(self._lat_hist))*1000.0
                max_ms = (max(self._lat_hist))*1000.0
                print(f"[VLM latency] {lat_ms:.1f} ms  (recent {len(self._lat_hist)} avg/min/max: "
                      f"{avg_ms:.1f}/{min_ms:.1f}/{max_ms:.1f} ms)")
                # parse suggestion
                self._last_vlm_vw = self.vlm.parse_action(out)

            suggested_vw = self._last_vlm_vw

            best_act, best_cost = (0.0, 0.0), float('inf')

            # 2) 遍历候选动作，用同一对 suggested_vw 打分（不再重复 query）
            for a in self.candidate_actions:
                v = clamp(float(a[0]), -self.v_max, self.v_max)
                w = clamp(float(a[1]), -self.w_max, self.w_max)

                C_goal = 0.2 * (abs(w)) + 0.05 * max(0.0, 0.2 - v)

                C_obst = 0.0
                if self.depth is not None and isinstance(self.depth, np.ndarray) and self.depth.size > 0:
                    d = self.depth
                    near_mask = (d > 0) & (d < 0.8)
                    frac_near = float(np.count_nonzero(near_mask)) / float(d.size)
                    C_obst = 1.0 * frac_near

                C_social = self.vlm.c_social((v, w), suggested_vw)

                C_total = self.alpha * C_goal + self.beta * C_obst + self.gamma * C_social
                if C_total < best_cost:
                    best_cost = C_total
                    best_act = (v, w)
            v_sel, w_sel = best_act
            v_h,  w_h    = self._last_vlm_vw
            pull = float(self.cfg.get("vlm_pull", 0.4))  # 0~1，小到中
            w_sel = clamp((1.0 - pull) * w_sel + pull * w_h, -self.w_max, self.w_max)
            best_act = (v_sel, w_sel)

            # 3) 保持/发布
            now = time.time()
            with self._cmd_lock:
                if now >= self._hold_until:
                    self._held_act = best_act
                    self._hold_until = now + self.action_duration_sec
                self._current_cmd = self._held_act

            self._vlm_tick += 1

            spent = time.time() - t0
            time.sleep(max(0.0, dt_plan - spent))

            print(f"[VLM suggest] v_h={self._last_vlm_vw[0]:+.2f}, w_h={self._last_vlm_vw[1]:+.2f}")
            print(f"[PLAN] pick v={best_act[0]:+.2f}, w={best_act[1]:+.2f}, total={best_cost:.3f}")

    def _publisher_loop(self):
        dt_pub = 1.0 / max(1e-3, self.publish_frequency)
        while True:
            with self._cmd_lock:
                v, w = self._current_cmd
            msg = {
                'linear':  {'x': float(v), 'y': 0.0, 'z': 0.0},
                'angular': {'x': 0.0, 'y': 0.0, 'z': float(w)},
            }
            try:
                self.cmd_pub.publish(msg)
            except Exception as e:
                print(f"[PUBLISH] {e}")
            time.sleep(dt_pub)


if __name__ == "__main__":
    node = VLMSocialNavNode()
    try:
        while True:
            time.sleep(1.0)  # keep main thread alive so daemon threads run
    except KeyboardInterrupt:
        pass
    finally:
        # Final latency summary
        try:
            if node.lat_count > 0:
                avg_lat = node.lat_sum / node.lat_count
                print(f"\n[VLM Latency Summary] calls={node.lat_count}, avg={avg_lat*1000:.1f} ms")
            else:
                print("\n[VLM Latency Summary] no VLM calls made.")
        except Exception as e:
            print(f"\n[VLM Latency Summary] error: {e}")
        # Cleanly terminate rosbridge connection
        try:
            if node.ros.is_connected:
                node.ros.terminate()
        except Exception:
            pass
