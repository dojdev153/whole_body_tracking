# Whole-Person Tracking & Face-Association Software Module
### CPU-Only Temporal Identity Continuity with an Honest Three-State Model

This software module implements an end-to-end computer vision and robotics pipeline that tracks whole persons, associates facial identities, maintains temporal continuity through temporary facial loss, and computes target centering controls for pan/tilt visual servoing without requiring physical hardware actuators.

---

## Key Architectural Components

### 1. Honest Three-State Model with Exponential Confidence Decay
Unlike naive tracking systems that either instantly forget an identity upon brief facial loss or falsely maintain 100% confidence indefinitely, this module implements an **Honest Three-State Model**:
- **CONFIRMED** (Green): Face is actively observed, mapped to the body, and verified against the ArcFace database. High confidence $C \in [0.60, 1.0]$.
- **TEMPORARY** (Amber/Gold): Face is temporarily lost (e.g., subject turns head 90–180°, looks down, or encounters partial occlusion), but the person body continues to be tracked by the Kalman filter. Identity is preserved while confidence decays exponentially:
  $$C(t) = C_0 \cdot e^{-\lambda \Delta t}$$
  - $C_0$: Initial confidence at the moment of facial loss.
  - $\lambda$: Decay rate parameter (default $\lambda = 0.50\text{ s}^{-1}$, yielding half-life $t_{1/2} \approx 1.38\text{ s}$).
  - $\Delta t$: Elapsed time since face loss ($t - t_{\text{last\_confirmed}}$).
  - *Re-acquisition*: If the face reappears within timeout $T_{\max} = 3.5\text{ s}$ and matches, state immediately restores to `CONFIRMED`.
- **UNKNOWN** (Red/Crimson): If confidence decays below floor $C_{\min} = 0.20$ or face loss exceeds $T_{\max}$, the track transparently transitions to `UNKNOWN`, acknowledging uncertainty until re-identified.

### 2. High-Performance CPU Vision Stack (~49 FPS)
- **Face Detection**: OpenCV YuNet ONNX (`models/face_detection_yunet_2023mar.onnx`) with 5 facial landmark regression (eyes, nose, mouth corners) in ~8 ms.
- **Person Detection**: MediaPipe Person Detection ONNX (`models/person_detection_mediapipe_2023mar.onnx`) with letterbox scaling and SSD anchor decoding in ~11 ms.
- **Feature Embedding**: ArcFace ONNX (`models/embedder_arcface.onnx`) with 5-point landmark similarity alignment producing L2-normalized 512-D identity vectors.
- **Multi-Object Tracking (MOT)**: 7-state Constant Velocity Kalman Filter ($[x_c, y_c, s, r, \dot{x}_c, \dot{y}_c, \dot{s}]$) with IoU Hungarian assignment.

### 3. Spatial Face-to-Body Mapping
- Solves the assignment problem between detected faces and detected person bodies.
- Enforces anatomical plausibility:
  - Bounding box containment ratio $\ge 25\%$.
  - Face vertical position within the top $50\%$ of the body bounding box.
  - Scale consistency ($w_{\text{face}} \le 1.25 \cdot w_{\text{body}}$).
  - Hungarian linear sum assignment on normalized spatial distance and inverse containment.
  - Close-up webcam extrapolation for subjects seated near the camera.

### 4. Target Centering & Virtual Gimbal Servoing
- Calculates normalized error offsets from optical center $(W/2, H/2)$:
  $$e_x = \frac{x_{\text{target}} - W/2}{W/2} \in [-1.0, 1.0], \quad e_y = \frac{y_{\text{target}} - H/2}{H/2} \in [-1.0, 1.0]$$
- Incorporates a $\pm 5\%$ deadband to prevent mechanical jitter.
- Integrates a Proportional-Derivative (PD) virtual gimbal simulator to validate pan/tilt tracking without physical hardware motors.

---

## Directory Structure

```
├── data/
│   ├── db/
│   │   ├── face_db.npz            # Enrolled ArcFace 512-D identity embeddings
│   │   └── face_db.json           # Identity names & metadata
│   └── scenarios/                 # Generated scenario clips and evaluation plots
│       ├── eval_head_turns_decay.png
│       ├── eval_target_centering.png
│       ├── eval_latency_breakdown.png
│       └── *.avi
├── models/
│   ├── embedder_arcface.onnx      # ArcFace 512-D ONNX model
│   ├── face_detection_yunet_2023mar.onnx
│   ├── person_detection_mediapipe_2023mar.onnx
│   └── mp_anchors.npy
├── src/
│   ├── association/               # Spatial face-to-body matching (Hungarian)
│   ├── camera.py                  # Physical external camera prioritization & stream handler
│   ├── control/                   # Target centering logic & virtual PTZ gimbal
│   ├── detectors/                 # Face detector, person detector, embedder, database
│   ├── eval/                      # Scenario generator & system evaluator
│   ├── tracker/                   # Kalman filter, Track, WholePersonTracker (3-state model)
│   └── visualization/             # Tactical HUD & aesthetic UI renderer
├── main.py                        # Interactive live webcam application
├── enroll_face.py                 # Dedicated face enrollment CLI utility
├── evaluate.py                    # Automated benchmark evaluation suite
└── EVALUATION_REPORT.md           # Scientific evaluation results & metrics
```

---

## Usage Instructions

### 1. Run Live Tracking with Webcam
```bash
# Automatically selects physical external USB camera over internal PC webcam:
python main.py

# Or specify a specific camera index or video file:
python main.py --cam 0
python main.py --cam data/scenarios/scenario_head_turns.avi

# Set a specific target identity to lock and center (default: 'Duff'):
python main.py --target Duff --decay 0.50 --timeout 3.5
```

**Interactive In-App Keyboard Controls**:
- `E` : Capture and enroll face in view with a custom identity name.
- `T` : Toggle tactical robotics HUD overlay.
- `C` : Reset simulated gimbal pan/tilt orientation to home $(0, 0)$.
- `Q` / `ESC` : Exit application.

### 2. Enroll New Face Identities
```bash
# Live interactive enrollment from webcam:
python enroll_face.py --name "Alice"

# Enroll from an existing image:
python enroll_face.py --name "Bob" --image path/to/photo.jpg

# List all enrolled identities in database:
python enroll_face.py --list

# Remove an identity:
python enroll_face.py --remove "Bob"
```

### 3. Run Automated Evaluation Suite
Tests real-world scenarios (head turns, dynamic occlusions, and path crossings) without requiring physical hardware:
```bash
python evaluate.py
```
This generates:
- Scenario test videos in `data/scenarios/`
- Visual performance plots (`eval_head_turns_decay.png`, `eval_target_centering.png`, `eval_latency_breakdown.png`)
- Comprehensive Markdown metrics report in `EVALUATION_REPORT.md`
