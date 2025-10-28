# 🧭 VLM-Social-Nav Tasks

This repository serves as a baseline implementation for Task-Oriented Visual-Semantic Social Navigation. It provides **five modular navigation nodes** (`task1_node.py` ~ `task5_node.py`) for social and object-goal navigation using **Vision-Language Models (VLMs)**.  
Each node handles a specific autonomous behavior (e.g., following a person, go to the front desk, or navigating to a pallet jack) and shares a **unified configuration file** (`config/params.yaml`) for parameter management.

---

## 📦 Repository Structure

```
vlm_social_nav/
├── task1_node.py         # Task 1: Follow person (VLM-guided)
├── task2_node.py         # Task 2: Social-aware navigation
├── task3_node.py         # Task 3: Obstacle-aware person following
├── task4_node.py         # Task 4: Go-to-front-desk navigation
├── task5_node.py         # Task 5: Go-to-pallet-jack (warehouse)
├── config/
│   └── params.yaml       # Unified configuration file (editable)
└── README.md             # This document
```

---

## ⚙️ Configuration (config/params.yaml)

All five nodes load parameters from the same YAML file.  
You can modify this file to tune navigation weights, topics, or timing without editing any Python code.

**Example:**
```bash
vlm_api_key: "sk-xxxx"
alpha: 0.3
beta: 1.0
gamma: 0.8
publish_frequency: 20.0
planner_frequency: 10.0
v_max: 0.5
w_max: 1.2
```


---

## 🧩 Key Parameters

| Parameter | Description | Default |
|------------|-------------|----------|
| `alpha, beta, gamma` | Weights for goal, obstacle, and social consistency terms | 0.3, 1.0, 0.8 |
| `wl, wa` | Weights for linear/angular consistency in VLM cost | 1.0, 1.0 |
| `v_max, w_max` | Velocity saturation limits | 0.5, 1.2 |
| `candidate_actions` | Action candidates `[v, w]` pairs | predefined list |
| `publish_frequency` | Rate for `/cmd_vel` publisher | 20 Hz |
| `planner_frequency` | Main control loop rate | 10 Hz |
| `segment_sec` | Duration of each VLM-guided action | 1.5 s |
| `vlm_period_sec` | Time interval between VLM queries | 4.0 s |
| `obs_stop_dist` | Emergency stop distance (depth) | 0.55 m |
| `vlm_api_key` | API key for OpenAI / compatible LLM | `""` or env var |

For the full parameter list, see [`config/params.yaml`](./config/params.yaml).

---

## 🚀 Running the Nodes

### 1. Install Dependencies
```bash
pip install roslibpy openai opencv-python pyyaml numpy
```

### 2. Start ROS Bridge
```bash
roslaunch rosbridge_server rosbridge_websocket.launch
```

### 3. Launch a Task
Each task connects to ROS topics and publishes velocity commands (`geometry_msgs/Twist`) based on camera inputs and VLM reasoning.

Example:
```bash
python3 task1_node.py   # Follow the person
python3 task5_node.py   # Go to the pallet jack
```

---

## 🧠 How It Works

Each node:
1. Subscribes to **RGB** and (optionally) **Depth** topics via `roslibpy`.
2. Calls a **Vision-Language Model (VLM)** (e.g., GPT-4o) using the current frame.
3. Parses the output text (e.g., “Move right with constant speed”).
4. Maps it to control commands (linear/angular velocity).
5. Combines it with obstacle and social cost functions.
6. Publishes `/cmd_vel` messages to control the robot.

---

## 🔧 Example Workflow

```bash
# Example run
export ROSLIBPY_HOST=192.168.0.10
export OPENAI_API_KEY="sk-your-key"
python3 task4_node.py
```

If you want to test with a different configuration:
```bash
VLM_SOCIAL_CFG=./config/warehouse.yaml python3 task5_node.py
```

---

## 🧩 Tasks Overview

| Task | Description | Notes |
|------|--------------|-------|
| **Task 1** | Follow a person in the view (hospital environment)
| **Task 2** | Navigate to a front desk (hospital environment)
| **Task 3** | Stay in the public area (hospital environment)
| **Task 4** | Go to the pallet jack without paying attention to safety markings (warehouse environment)
| **Task 5** | Go to the pallet jack and pay attention to safety markings (warehouse environment)

---





