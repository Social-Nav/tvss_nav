#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VLM-Go-To-Pallet-Jack (warehouse, single file, threaded) — VLM-only + decoupled timing
- Task: Navigate to the PALLET JACK (托盘搬运车) in a warehouse
- All decisions by VLM from camera image only (no jack_offset/scale)
- Depth-based obstacle avoidance stays ON
- Decoupled timing:
    * VLM query every vlm_period_sec seconds (e.g., 2.0s)
    * Execute each suggestion for segment_sec seconds (e.g., 1.0s), then COOLDOWN (stop) until next period
- VLM Answer format enforced: "Move DIRECTION with SPEED"
  (DIRECTION ∈ {left, straight, right}; SPEED ∈ {slow down, constant, speed up, stop})

Connections (via rosbridge):
  /camera/color/image_raw/compressed  (sensor_msgs/CompressedImage)
  /camera/depth/image_raw             (sensor_msgs/Image)        [optional]
  /tf, /tf_static                     (tf2_msgs/TFMessage)
  /cmd_vel                            (geometry_msgs/Twist)

Config (./config/params.yaml or env VLM_SOCIAL_CFG):
  alpha, beta, gamma     : weights for C_goal, C_obst, C_social   [C_goal 仅做轻微直行/小角速度正则]
  wl, wa                 : weights inside C_social
  publish_frequency, planner_frequency
  candidate_actions      : [[v,w], ...]
  v_max, w_max           : saturation
  vlm_pull               : final "velocity attractor" towards VLM suggestion in ω
  segment_sec            : seconds to execute each suggestion (e.g., 1.0)
  vlm_period_sec         : period between two VLM calls (e.g., 2.0)  ← 新增

  # --- obstacle avoidance (depth-based) ---
  obs_stop_dist          : emergency stop distance (m), default 0.55
  obs_warn_dist          : speed-limit start distance (m), default 1.00
  obs_roi_y0             : depth ROI start (relative height 0~1), default 0.70
  obs_max_depth          : clamp depth values to this maximum (m), default 4.0
  obs_near_frac_gain     : occupancy-cost gain, default 1.2
  obs_turn_bias_gain     : lateral turn bias gain, default 0.25

  cmd_vel_topic          : string
  vlm_api_key            : string or use env OPENAI_API_KEY
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

# ----------------- VLM cost scorer (Image-only) -----------------
class VLMCostScorer:
    def __init__(self, api_key, wl=0.7, wa=1.0):
        self.wl = float(wl)
        self.wa = float(wa)
        self.base_v = 0.40
        self.dir_to_w = {"left": +0.20, "straight": 0.0, "right": -0.20}
        self.spd_to_v = {"slow down": 0.10, "constant": 0.40, "speed up": 0.50, "stop": 0.00}
        self.prompt_template = (
            "Task: You are in a warehouse. Go to the orange PALLET JACK (manual pallet truck) in the scene.\n"
            "Rules:\n"
            "- Make decisions from the camera image only.\n"
            "- Ignore floor labels/markings, except for yellow warning areas.\n"
            "- Do NOT enter yellow warning areas (treat them as obstacles).\n"
            "- Steer toward the pallet jack if it appears left/right.\n"
            "- Slow down as you get close; say 'stop' at the pallet jack.\n"
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

    def _build_messages(self, image_bgr):
        content = [{"type": "text", "text": self.prompt_template}]
        if image_bgr is not None:
            ok, buf = cv2.imencode(".jpg", image_bgr)
            if ok:
                b64 = base64.b64encode(buf).decode("utf-8")
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        return [{"role": "user", "content": content}]

    def query(self, image_bgr):
        messages = self._build_messages(image_bgr)
        try:
            if self.use_v1:
                resp = self.client.chat.completions.create(
                    model=("gpt-4o" if image_bgr is not None else "gpt-4o-mini"),
                    messages=messages, max_tokens=48, temperature=0.2,
                )
                out = resp.choices[0].message.content.strip()
            else:
                resp = openai.ChatCompletion.create(
                    model=("gpt-4-vision-preview" if image_bgr is not None else "gpt-4o-mini"),
                    messages=messages, max_tokens=48, temperature=0.2,
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
        w = {"left": +0.14, "straight": 0.0, "right": -0.14}.get(direction, 0.0)
        return v, w

    def c_social(self, action_vw, suggested_vw):
        v, w = action_vw
        v_h, w_h = suggested_vw
        return self.wl * abs(v - v_h) + self.wa * abs(w - w_h)

# ----------------- Main Node -----------------
class VLMGoToPalletJackNode:
    def __init__(self):
        # ROS
        host = os.environ.get("ROSLIBPY_HOST", "localhost")
        port = int(os.environ.get("ROSLIBPY_PORT", "9090"))
        self.ros = Ros(host, port); self.ros.run()
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
        self.depth_topic = self.cfg.get("depth_topic", "/camera/aligned_depth_to_color/image_raw")
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
        self.alpha = float(self.cfg.get("alpha", 0.1))   # 轻量直行/小角速度正则
        self.beta  = float(self.cfg.get("beta", 1.0))    # 避障权重
        self.gamma = float(self.cfg.get("gamma", 0.7))   # VLM一致性
        self.wl    = float(self.cfg.get("wl", 0.7))
        self.wa    = float(self.cfg.get("wa", 1.0))
        self.v_max = float(self.cfg.get("v_max", 0.35))
        self.w_max = float(self.cfg.get("w_max", 1.0))

        # Action set & rates
        self.candidate_actions = self.cfg.get("candidate_actions", [
            [0.00, +0.50], [0.10, +0.25], [0.18, +0.12],
            [0.18,  0.00],
            [0.18, -0.12], [0.10, -0.25], [0.00, -0.50],
            [0.00,  0.00]
        ])
        self.publish_frequency = float(self.cfg.get("publish_frequency", 12.0))
        self.planner_frequency = float(self.cfg.get("planner_frequency", 8.0))

        # --- Timing (decoupled) ---
        self.segment_sec = float(self.cfg.get("segment_sec", 1.5))
        self.vlm_period_sec = float(self.cfg.get("vlm_period_sec", 4.0))  # 新增：VLM 调用周期
        self._t_last_vlm = 0.0
        self._cooldown_until = 0.0

        # VLM module
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

        # Locks / command state
        self._cmd_lock = threading.Lock()
        self._current_cmd = (0.0, 0.0)
        self._held_act = (0.0, 0.0)
        self._hold_until = 0.0

        # State machine: WAIT / EXEC / COOLDOWN
        self._state = "WAIT"
        self._last_vlm_vw = (0.15, 0.0)

        # --- obstacle avoidance params ---
        self.obs_stop_dist = float(self.cfg.get("obs_stop_dist", 0.55))
        self.obs_warn_dist = float(self.cfg.get("obs_warn_dist", 1.00))
        self.obs_roi_y0    = float(self.cfg.get("obs_roi_y0", 0.70))
        self.obs_max_depth = float(self.cfg.get("obs_max_depth", 4.0))
        self.obs_near_frac_gain = float(self.cfg.get("obs_near_frac_gain", 1.2))
        self.obs_turn_bias_gain = float(self.cfg.get("obs_turn_bias_gain", 0.25))

        # Threads
        threading.Thread(target=self._planner_loop, daemon=True).start()
        threading.Thread(target=self._publisher_loop, daemon=True).start()
        print(f"[Node] VLM-Go-To-Pallet-Jack ready. Publishing {self.cmd_vel_topic}")

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
                "alpha": 0.1, "beta": 1.0, "gamma": 0.7,
                "publish_frequency": 12.0,
                "planner_frequency": 8.0,
                "wl": 0.7, "wa": 1.0,
                "candidate_actions": [[0.20,0.0],[0.20,0.12],[0.20,-0.12],[0.10,0.0],[0.0,0.0]],
                "vlm_api_key": os.environ.get("OPENAI_API_KEY",""),
                "cmd_vel_topic": "/jackal/cmd_vel",
                "vlm_pull": 0.50,
                "segment_sec": 1.0,
                "vlm_period_sec": 4.0,     # 默认：每 2s 调用一次 VLM
                # defaults for obstacle avoidance
                "obs_stop_dist": 0.55,
                "obs_warn_dist": 1.00,
                "obs_roi_y0": 0.70,
                "obs_max_depth": 4.0,
                "obs_near_frac_gain": 1.2,
                "obs_turn_bias_gain": 0.25,
                "v_max": 0.35,
                "w_max": 1.0,
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

    # ---------- Depth sectors helper (L/C/R) ----------
    def _depth_sectors(self, dimg):
        """
        Returns:
          {'L': {'d_min', 'frac_near'}, 'C': {...}, 'R': {...},
           'front_min': float, 'near_any': bool}
        """
        H, W = dimg.shape[:2]
        y0 = int(H * self.obs_roi_y0)
        roi = dimg[y0:H, :]
        roi = np.clip(roi, 0.0, self.obs_max_depth)
        valid = roi > 0

        thirds = np.array_split(roi, 3, axis=1)
        thirds_val = np.array_split(valid, 3, axis=1)

        out = {}
        front_min = float('inf')
        near_any = False

        for name, seg, segv in zip(['L','C','R'], thirds, thirds_val):
            if segv.any():
                d_min = float(seg[segv].min())
                near_mask = segv & (seg < self.obs_warn_dist) & (seg > 0)
                frac_near = float(near_mask.sum()) / float(seg.size)
            else:
                d_min = float('inf')
                frac_near = 0.0
            out[name] = {'d_min': d_min, 'frac_near': frac_near}
            if name == 'C':
                front_min = d_min
            if d_min < self.obs_warn_dist:
                near_any = True

        out['front_min'] = front_min
        out['near_any']  = near_any
        return out

    # ------------- Planner & Publisher (decoupled timing) -------------
    def _planner_loop(self):
        dt_plan = 1.0 / max(1e-3, self.planner_frequency)
        while True:
            t0 = time.time()
            if self.rgb is None:
                time.sleep(dt_plan); continue

            # Depth sectors (for obstacle avoidance)
            if self.depth is not None and isinstance(self.depth, np.ndarray) and self.depth.size > 0:
                sector = self._depth_sectors(self.depth)
            else:
                sector = {'L': {'d_min': float('inf'), 'frac_near': 0.0},
                          'C': {'d_min': float('inf'), 'frac_near': 0.0},
                          'R': {'d_min': float('inf'), 'frac_near': 0.0},
                          'front_min': float('inf'), 'near_any': False}

            now = time.time()

            # ================== EXEC ==================
            if self._state == "EXEC":
                if now < self._hold_until:
                    # Emergency stop during EXEC
                    if sector and sector['front_min'] < self.obs_stop_dist:
                        with self._cmd_lock:
                            self._held_act = (0.0, 0.0)
                            self._current_cmd = (0.0, 0.0)
                            self._hold_until = 0.0
                        self._cooldown_until = max(self._t_last_vlm + self.vlm_period_sec, now)
                        self._state = "COOLDOWN"
                        print(f"[SAFETY] EXEC emergency stop -> COOLDOWN {self._cooldown_until-now:.2f}s")
                else:
                    # segment finished -> stop and COOLDOWN
                    with self._cmd_lock:
                        self._held_act = (0.0, 0.0)
                        self._current_cmd = (0.0, 0.0)
                        self._hold_until = 0.0
                    self._cooldown_until = max(self._t_last_vlm + self.vlm_period_sec, now)
                    self._state = "COOLDOWN"
                    print(f"[SEGMENT] done -> COOLDOWN {self._cooldown_until-now:.2f}s")
                time.sleep(max(0.0, dt_plan - (time.time()-t0)))
                continue

            # ================== COOLDOWN ==================
            if self._state == "COOLDOWN":
                # stay stopped until next cycle boundary
                if now >= self._cooldown_until:
                    self._state = "WAIT"
                    print("[COOLDOWN] ended -> WAIT")
                time.sleep(max(0.0, dt_plan - (time.time()-t0)))
                continue

            # ================== WAIT ==================
            if self._state == "WAIT":
                # Only call VLM when reached next period boundary
                if (now - self._t_last_vlm) < self.vlm_period_sec:
                    # Not yet time: ensure stop
                    with self._cmd_lock:
                        self._held_act = (0.0, 0.0)
                        self._current_cmd = (0.0, 0.0)
                    time.sleep(max(0.0, dt_plan - (time.time()-t0)))
                    continue

                # Time to call VLM
                out = self.vlm.query(self.rgb)
                self._last_vlm_vw = self.vlm.parse_action(out)
                v_h, w_h = self._last_vlm_vw
                self._t_last_vlm = now
                print(f"[VLM suggest] v_h={v_h:+.2f}, w_h={w_h:+.2f} (period={self.vlm_period_sec:.2f}s)")

                # --- Candidate scoring ---
                best_act, best_cost = (0.0, 0.0), float('inf')
                for a in self.candidate_actions:
                    v = clamp(float(a[0]), -self.v_max, self.v_max)
                    w = clamp(float(a[1]), -self.w_max, self.w_max)

                    # Light 'posture regularizer': prefer straight & small |w|
                    C_goal = 0.05*abs(w) + 0.02*max(0.0, 0.18 - v)

                    # Obstacle term (depth-based)
                    C_obst = 0.0
                    if sector:
                        occ_cost = (sector['L']['frac_near'] + sector['C']['frac_near'] + sector['R']['frac_near']) / 3.0
                        C_obst += self.obs_near_frac_gain * occ_cost
                        if v > 0.0 and np.isfinite(sector['front_min']):
                            front_pen = max(0.0, (self.obs_warn_dist - sector['front_min'])) / max(1e-3, self.obs_warn_dist)
                            C_obst += 1.2 * front_pen * v

                    # Social/VLM alignment
                    C_social = self.vlm.c_social((v, w), (v_h, w_h))

                    C_total = self.alpha * C_goal + self.beta * C_obst + self.gamma * C_social
                    if C_total < best_cost:
                        best_cost = C_total; best_act = (v, w)

                # VLM pull on ω, and take v from VLM
                v_sel, w_sel = best_act
                pull = float(self.cfg.get("vlm_pull", 0.50))
                w_sel = clamp((1.0 - pull) * w_sel + pull * w_h, -self.w_max, self.w_max)
                v_sel = v_h

                # --- Safety / proximity scaling before EXEC ---
                dmin = sector['front_min']
                if dmin < self.obs_stop_dist:
                    # Too close: skip EXEC, go to cooldown
                    with self._cmd_lock:
                        self._held_act = (0.0, 0.0)
                        self._current_cmd = (0.0, 0.0)
                        self._hold_until = 0.0
                    self._cooldown_until = max(self._t_last_vlm + self.vlm_period_sec, now)
                    self._state = "COOLDOWN"
                    print(f"[SAFETY] too close -> skip EXEC, COOLDOWN {self._cooldown_until-now:.2f}s")
                    time.sleep(max(0.0, dt_plan - (time.time()-t0)))
                    continue
                elif dmin < self.obs_warn_dist:
                    scale = (dmin - self.obs_stop_dist) / max(1e-3, (self.obs_warn_dist - self.obs_stop_dist))
                    scale = clamp(scale, 0.0, 1.0)
                    v_sel = clamp(v_sel * scale, -self.v_max, self.v_max)
                    if scale < 0.35:
                        w_sel = clamp(w_sel * (0.6 + 0.4*scale), -self.w_max, self.w_max)

                # Lateral turn bias away from closer side
                dL, dR = sector['L']['d_min'], sector['R']['d_min']
                if (dL < float('inf')) and (dR < float('inf')):
                    sign = +1.0 if dR < dL else -1.0  # if right is closer, steer left (+w)
                    closeness = 1.0 / max(0.2, min(dL, dR))
                    frac_gap  = abs(sector['L']['frac_near'] - sector['R']['frac_near'])
                    delta_w = self.obs_turn_bias_gain * sign * (0.4*closeness + 0.6*frac_gap)
                    w_sel = clamp(w_sel + delta_w, -self.w_max, self.w_max)

                # Enter EXEC: execute for segment_sec seconds, then cooldown until next period
                with self._cmd_lock:
                    self._held_act = (v_sel, w_sel)
                    self._current_cmd = self._held_act
                    self._hold_until = time.time() + self.segment_sec
                self._state = "EXEC"
                print(f"[PLAN] v={v_sel:+.2f}, w={w_sel:+.2f}, run {self.segment_sec:.2f}s → cooldown "
                      f"{max(0.0, self.vlm_period_sec - self.segment_sec):.2f}s")

                time.sleep(max(0.0, dt_plan - (time.time()-t0)))
                continue

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
    node = VLMGoToPalletJackNode()
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
