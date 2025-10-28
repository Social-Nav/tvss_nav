#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VLM-Go-To-Front-Desk (article-aligned, single file, threaded)
- Task: Navigate to the FRONT DESK (reception counter)
- Lightweight geometric cue ("desk" = long horizontal countertop edge)
- Perception-gated VLM queries; high-rate cmd publisher; low-rate planner
- Answer format enforced: "Move DIRECTION with SPEED"
  (DIRECTION ∈ {left, straight, right}; SPEED ∈ {slow down, constant, speed up, stop})

Connections (via rosbridge):
  /camera/color/image_raw/compressed  (sensor_msgs/CompressedImage)
  /camera/depth/image_raw             (sensor_msgs/Image)        [optional]
  /tf, /tf_static                     (tf2_msgs/TFMessage)
  /cmd_vel                            (geometry_msgs/Twist)

Config (./config/params.yaml or env VLM_SOCIAL_CFG):
  alpha, beta, gamma     : weights for C_goal, C_obst, C_social
  wl, wa                 : weights inside C_social
  publish_frequency, planner_frequency
  action_duration_sec
  candidate_actions      : [[v,w], ...]
  vlm_every_n            : call VLM every N planner ticks (>=1)
  use_vlm_only_when_social: bool; gate VLM on desk cue
  yolo_conf              : kept for compatibility (unused here)
  v_max, w_max           : saturation
  vlm_pull               : final "velocity attractor" towards VLM suggestion in ω
  goal_stop_scale        : stop when desk_scale ≥ this value (goal reached)
  goal_slow_scale        : begin slowing down above this value
  cmd_vel_topic          : string
  vlm_api_key            : string or use env OPENAI_API_KEY

Notes:
- No color-based detection; desk cue from edges + Hough lines.
- C_goal encourages (i) reduce |desk_offset|, (ii) increase desk_scale until stop.
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

# ----------------- Perception: FRONT DESK cue -----------------
class FrontDeskPerception:
    """
    Geometry-only cue for a 'front desk' / reception counter:
      - detect long near-horizontal edges (countertop) via Canny + HoughLinesP
      - prefer lines in lower-middle region; filter by aspect and angle
    Output:
      - has_desk: bool
      - desk_offset in [-1, 1]  (line midpoint horizontal offset: left negative, right positive)
      - desk_scale  in [0, 1]   (proxy for approach: normalized line length × proximity weighting)
    """
    def __init__(self):
        self.last = (False, 0.0, 0.0)

    def get_desk_geo(self, bgr_image):
        if bgr_image is None:
            return self.last
        H, W = bgr_image.shape[:2]
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

        # Robust edges without color reliance
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        edges = cv2.Canny(blur, 60, 160, L2gradient=True)

        # Only look at lower 60% region (reception counters usually below midline)
        roi_y0 = int(H * 0.40)
        mask = np.zeros_like(edges)
        mask[roi_y0:H, :] = 255
        edges = cv2.bitwise_and(edges, mask)

        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180,
                                threshold=60, minLineLength=int(W*0.25), maxLineGap=12)
        if lines is None:
            self.last = (False, 0.0, 0.0); return self.last

        best_score = -1.0
        best_mid = (W*0.5, H*0.7)
        best_len = 0.0
        for l in lines[:,0,:]:
            x1,y1,x2,y2 = map(int, l)
            dx, dy = (x2-x1), (y2-y1)
            length = (dx*dx + dy*dy) ** 0.5
            if length < W*0.22:   # ensure reasonably long
                continue
            angle = abs(np.degrees(np.arctan2(dy, dx)))
            angle = min(angle, 180-angle)  # 0..90
            if angle > 12.0:  # near-horizontal only
                continue
            midx = 0.5*(x1+x2)
            midy = 0.5*(y1+y2)
            # Prefer lines closer to image center horizontally and lower vertically (closer to robot)
            center_bias = 1.0 - min(1.0, abs(midx - W*0.5)/(W*0.5))
            vertical_bias = min(1.0, (midy - roi_y0) / (H - roi_y0 + 1e-6))
            score = 0.50*center_bias + 0.35*vertical_bias + 0.15*(length/(W+1e-6))
            if score > best_score:
                best_score = score
                best_mid = (midx, midy)
                best_len = length

        if best_score < 0.0:
            self.last = (False, 0.0, 0.0); return self.last

        offset = (best_mid[0] - W*0.5) / (W*0.5)
        # scale: line length normalized with small boost if lower in image
        row_factor = (best_mid[1] - roi_y0) / (H - roi_y0 + 1e-6)
        row_factor = clamp(row_factor, 0.0, 1.0)
        scale = clamp((best_len / W) * (0.65 + 0.35*row_factor), 0.0, 1.0)

        has = True
        self.last = (has, float(offset), float(scale))
        return self.last

# ----------------- VLM cost scorer (Front Desk) -----------------
class VLMCostScorer:
    def __init__(self, api_key, wl=0.6, wa=1.0):
        self.wl = float(wl)
        self.wa = float(wa)
        # conservative base velocities
        self.base_v = 0.30
        self.dir_to_w = {"left": +0.15, "straight": 0.0, "right": -0.15}
        self.spd_to_v = {
            "slow down": 0.12,
            "constant":  0.30,
            "speed up":  0.35,
            "stop":      0.00,
        }
        # Prompt for FRONT DESK
        self.prompt_template = (
            "Task: Go to the FRONT DESK (reception counter) visible in the scene.\n"
            "Behaviors:\n"
            "- Steer toward the counter if it is offset left/right.\n"
            "- Slow down as you get close; stop at the counter.\n"
            "- Keep motion smooth and avoid abrupt turns.\n\n"
            "Ego state:\n"
            "- heading: {heading}\n"
            "- linear velocity: {speed:.2f}\n"
            "- desk_offset: {desk_offset:.2f}  # [-1..1], left negative, right positive\n"
            "- desk_scale:  {desk_scale:.3f}   # [0..1], larger means closer/larger counter\n\n"
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

    def _build_messages(self, heading, speed, image_bgr=None, desk_offset=0.0, desk_scale=0.0):
        content = [{
            "type": "text",
            "text": self.prompt_template.format(
                heading=heading, speed=speed,
                desk_offset=desk_offset, desk_scale=desk_scale,
            )
        }]
        if image_bgr is not None:
            ok, buf = cv2.imencode(".jpg", image_bgr)
            if ok:
                b64 = base64.b64encode(buf).decode("utf-8")
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        return [{"role": "user", "content": content}]

    def query(self, heading="straight", speed=0.20, image_bgr=None, desk_offset=0.0, desk_scale=0.0):
        if image_bgr is not None:
            try:
                cv2.imshow("VLM Input", image_bgr)
                cv2.waitKey(1)
            except Exception:
                pass
        messages = self._build_messages(heading, speed, image_bgr, desk_offset, desk_scale)
        try:
            if self.use_v1:
                resp = self.client.chat.completions.create(
                    model=("gpt-4o" if image_bgr is not None else "gpt-4o-mini"),
                    messages=messages,
                    max_tokens=48,
                    temperature=0.2,
                )
                out = resp.choices[0].message.content.strip()
            else:
                resp = openai.ChatCompletion.create(
                    model=("gpt-4-vision-preview" if image_bgr is not None else "gpt-4o-mini"),
                    messages=messages,
                    max_tokens=48,
                    temperature=0.2,
                )
                out = resp["choices"][0]["message"]["content"].strip()
            print(f"[VLM Output] {out}")
            return out
        except Exception as e:
            print(f"[VLM ERROR] {e}")
            return "Move straight with constant"

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
        # keep as "alignment cost to VLM suggestion"
        v, w = action_vw
        v_h, w_h = suggested_vw
        return self.wl * abs(v - v_h) + self.wa * abs(w - w_h)

# ----------------- Main Node -----------------
class VLMGoToFrontDeskNode:
    def __init__(self):
        # ROS
        host = os.environ.get("ROSLIBPY_HOST", "localhost")
        port = int(os.environ.get("ROSLIBPY_PORT", "9090"))
        self.ros = Ros(host, port)
        self.ros.run()
        while not self.ros.is_connected:
            time.sleep(0.02)
        print(f"[ROS] Connected {host}:{port}")

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
        self.alpha = float(self.cfg.get("alpha", 1.0))   # goal term (desk approach)
        self.beta  = float(self.cfg.get("beta", 1.0))    # obstacle term
        self.gamma = float(self.cfg.get("gamma", 0.6))   # vlm alignment
        self.wl    = float(self.cfg.get("wl", 0.6))
        self.wa    = float(self.cfg.get("wa", 1.0))
        self.v_max = float(self.cfg.get("v_max", 0.45))  # slower overall
        self.w_max = float(self.cfg.get("w_max", 1.0))

        # Goal approach thresholds
        self.goal_stop_scale = float(self.cfg.get("goal_stop_scale", 0.5))
        self.goal_slow_scale = float(self.cfg.get("goal_slow_scale", 0.5))

        # Action set & rates (conservative)
        self.candidate_actions = self.cfg.get("candidate_actions", [
            [0.00, +0.60], [0.12, +0.30], [0.22, +0.15],
            [0.22,  0.00],
            [0.22, -0.15], [0.12, -0.30], [0.00, -0.60],
            [0.00,  0.00]
        ])
        self.publish_frequency = float(self.cfg.get("publish_frequency", 1.0))
        self.planner_frequency = float(self.cfg.get("planner_frequency", 10.0))
        self.action_duration_sec = float(self.cfg.get("action_duration_sec", 0.5))

        # VLM gating
        self.vlm_every_n = int(self.cfg.get("vlm_every_n", 2))
        self._vlm_tick = 0
        self._last_vlm_vw = (0.20, 0.0)
        self._use_gate = bool(self.cfg.get("use_vlm_only_when_social", True))

        # Locks / command state
        self._cmd_lock = threading.Lock()
        self._current_cmd = (0.0, 0.0)
        self._held_act = (0.0, 0.0)
        self._hold_until = 0.0

        # --- VLM freshness & logging throttling ---
        self.require_fresh_vlm = bool(self.cfg.get("require_fresh_vlm", True))
        self.vlm_min_period = float(self.cfg.get("vlm_min_period", 0.5))   # 两次VLM最小间隔(s)
        self.vlm_timeout = float(self.cfg.get("vlm_timeout", 1))         # 允许旧建议的最大时长(s)
        self.on_stale_policy = self.cfg.get("on_stale_policy", "stop")     # "stop" | "crawl"
        self.crawl_speed = float(self.cfg.get("crawl_speed", 0.10))

        self._last_vlm_time = 0.0
        self._last_plan_log_t = 0.0
        self._last_suggest_log_t = 0.0
        self.log_throttle = float(self.cfg.get("log_throttle", 0.8))        # 打印节流间隔(s)

        # Modules
        self.perception = FrontDeskPerception()
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

        # Threads
        threading.Thread(target=self._planner_loop, daemon=True).start()
        threading.Thread(target=self._publisher_loop, daemon=True).start()
        print(f"[Node] VLM-Go-To-Front-Desk ready. Publishing {self.cmd_vel_topic}")

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
                "alpha": 1.0, "beta": 1.0, "gamma": 0.6,
                "publish_frequency": 20.0,
                "planner_frequency": 8.0,
                "wl": 0.6, "wa": 1.0,
                "candidate_actions": [[0.22,0.0],[0.22,0.15],[0.22,-0.15],[0.12,0.0],[0.0,0.0]],
                "vlm_api_key": os.environ.get("OPENAI_API_KEY",""),
                "cmd_vel_topic": "/jackal/cmd_vel",
                "goal_stop_scale": 0.16,
                "goal_slow_scale": 0.10,
                "vlm_pull": 0.45
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

    # ------------- Planner & Publisher -------------
    def _planner_loop(self):
        dt_plan = 1.0 / max(1e-3, self.planner_frequency)
        while True:
            t0 = time.time()
            if self.rgb is None:
                time.sleep(dt_plan); continue

            # 1) Perception (desk cue)
            has_desk, desk_offset, desk_scale = self.perception.get_desk_geo(self.rgb)
            if has_desk:
                print(f"[Perception] desk: offset={desk_offset:+.2f}, scale={desk_scale:.3f}")

            # 2) VLM gating: only when desk cue present (or disabled)
            need_vlm = (not self._use_gate) or has_desk
            call_vlm = (self._vlm_tick % max(1, self.vlm_every_n) == 0)
            if need_vlm and call_vlm:
                out = self.vlm.query(
                    heading="straight",
                    speed=0.20,
                    image_bgr=self.rgb,
                    desk_offset=(desk_offset if has_desk else 0.0),
                    desk_scale=(desk_scale if has_desk else 0.0),
                )
                self._last_vlm_vw = self.vlm.parse_action(out)

            suggested_vw = self._last_vlm_vw

            # 3) Candidate scoring
            best_act, best_cost = (0.0, 0.0), float('inf')
            for a in self.candidate_actions:
                v = clamp(float(a[0]), -self.v_max, self.v_max)
                w = clamp(float(a[1]), -self.w_max, self.w_max)

                # --- Goal term: center the desk & approach then stop ---
                # penalize heading away from desk_offset; prefer small |offset|
                goal_off = abs(desk_offset) if has_desk else 0.5  # if no cue, mild penalty
                # encourage approach (larger scale), but reduce v when near desk
                near_bonus = (1.0 - clamp(desk_scale, 0.0, 1.0))
                C_goal = 0.8*goal_off + 0.2*near_bonus + 0.10*abs(w)

                # If near desk, prefer low linear speed
                if has_desk and desk_scale >= self.goal_slow_scale:
                    C_goal += 0.8 * max(0.0, v - 0.12)  # prefer v<=0.12 near goal

                # If desk reached, prefer stop
                if has_desk and desk_scale >= self.goal_stop_scale:
                    C_goal += 2.0 * v   # strongly penalize non-zero v

                # --- Obstacle term: quick-and-dirty depth occupancy ---
                C_obst = 0.0
                if self.depth is not None and isinstance(self.depth, np.ndarray) and self.depth.size > 0:
                    d = self.depth
                    near_mask = (d > 0) & (d < 0.8)
                    frac_near = float(np.count_nonzero(near_mask)) / float(d.size)
                    C_obst = 1.2 * frac_near
                    # discourage forward speed when many near obstacles
                    C_obst += 0.6 * v * frac_near

                # --- Social/VLM alignment ---
                C_social = self.vlm.c_social((v, w), suggested_vw)

                C_total = self.alpha * C_goal + self.beta * C_obst + self.gamma * C_social
                if C_total < best_cost:
                    best_cost = C_total
                    best_act = (v, w)

            v_sel, w_sel = best_act
            v_h,  w_h    = self._last_vlm_vw
            pull = float(self.cfg.get("vlm_pull", 0.45))  # pull ω mildly toward VLM
            w_sel = clamp((1.0 - pull) * w_sel + pull * w_h, -self.w_max, self.w_max)
            best_act = (v_h, w_sel)

            # 4) Hold & publish
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
            print(f"[PLAN] pick v={best_act[0]:+.2f}, w={best_act[1]:+.2f}, total={best_cost:.3f}, desk_scale={desk_scale:.3f}")

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

    # ---------------- Utilities (unchanged) ----------------
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

if __name__ == "__main__":
    node = VLMGoToFrontDeskNode()
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if node.ros.is_connected:
                node.ros.terminate()
        except Exception:
            pass



