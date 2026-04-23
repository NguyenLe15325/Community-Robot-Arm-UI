# Community Robot Arm UI

A professional desktop GUI for controlling a 3-DOF robot arm over serial (G-code), with integrated computer vision for ArUco-based calibration and YOLO-powered automated sorting.

Built with **PyQt6** · **OpenCV** · **Python 3.10+**

### Related Repositories

| Repository | Description |
|------------|-------------|
| [Community-Robot-Arm](https://github.com/NguyenLe15325/Community-Robot-Arm) | Firmware — Arduino-based 3-DOF arm with G-code interpreter, kinematics, and stepper control |
| [Candy-Sorting](https://github.com/NguyenLe15325/Candy-Sorting) | YOLO models — trained weights for candy detection/sorting (see supported model types below) |

---

## Features

### 🎮 Control Tab
- **Cartesian Move** — absolute/incremental in robot or world frame
- **Joint Move** — direct theta1/theta2/theta3 control
- **Jog** — step-based XYZ jogging with configurable step size
- **Gripper** — open/close/home with adjustable distance and speed
- **World Frame Transform** — 6-DOF offset (translation + rotation) from world to robot coordinates
- **Teach Points** — save/recall named positions with one-click Go To and Insert into Program

### 👁️ Vision Tab
- **Live Camera Feed** — ArUco marker detection with configurable ROI
- **Frame View Toggle** — show ROI only, original only, or both
- **4-Corner Calibration** — map pixel coordinates to world coordinates using ArUco markers
- **Click-to-Pick** — click on the ROI frame to set pick XY in world coordinates
- **Move / Jog** — vision-local move and jog controls
- **Gripper** — vision-local gripper controls
- **Place Positions** — named place targets with save/load to JSON
- **Pick & Place Sequence** — G-code template editor with variable substitution (`{PICK_X}`, `{PLACE_Y}`, etc.)
- **Template Save/Load** — export/import PnP templates as `.gcode` files independently
- **YOLO Detection** — load a `.pt` model for real-time object detection. Supported model types:
  - **Object Detection** (`yolov8n.pt`, etc.) — bounding-box detection
  - **Instance Segmentation** (`yolov8n-seg.pt`, etc.) — pixel-level masks
  - **Oriented Bounding Box** (`yolov8n-obb.pt`, etc.) — rotated bounding boxes
  - Pre-trained candy sorting models available at [Candy-Sorting](https://github.com/NguyenLe15325/Candy-Sorting)
- **Auto Sort** — continuous Look-Pick-Look cycle with class-based sorting, idle parking, and wake-on-detection
- **Collapsible Sections** — toggle visibility of each group to reduce scrolling

### 📝 Program Tab
- **G-code Editor** — write multi-line programs with syntax guidance
- **Subroutines** — `CALL path/to/file.gcode [repeat]` with nested CALL support
- **Infinite Loop** — `CALL file.gcode 0` runs until Stop
- **Preamble Split** — `---` marker separates one-time setup from looped body
- **Loop Count** — repeat the entire program N times or infinitely
- **Record Mode** — toggle to capture manual commands into the program editor
- **Load / Save** — persist programs as `.gcode` files

### 🖥️ Terminal Tab
- **Raw Serial Console** — send arbitrary G-code commands
- **Command History** — Up/Down arrow navigation through previous commands

### ⚡ Sidebar (Always Visible)
- **Connection** — serial port selection, baud rate, connect/disconnect
- **Status** — live robot pose (robot + world), execution queue depth
- **Speed Override** — slider to scale all feedrates (1–100%)
- **Quick Commands** — M17 (enable), M18 (disable), G28 (home), M114 (position), M119 (endstops)
- **M112 Emergency Stop** — red button + `Escape` key shortcut (global)

### 💾 Settings Persistence
- All UI state auto-saved on close to `~/Documents/Community-Robot-Arm-UI/ui_settings.json`
- Includes: connection, vision calibration, teach points, gripper settings, detection config
- Save/Load settings to custom files via Vision tab buttons

---

## Quick Start

### Prerequisites
- Python 3.10 or higher
- The robot arm connected via USB serial (firmware: [Community-Robot-Arm](https://github.com/NguyenLe15325/Community-Robot-Arm))

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd Community-Robot-Arm-UI

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Optional: YOLO Detection
```bash
pip install ultralytics
```

### Run

```bash
python main.py
```

---

## G-code Reference

### Motion Commands

| Command | Description | Example |
|---------|-------------|---------|
| `G0 T<θ1> T<θ2> T<θ3>` | Joint move (degrees × 10) | `G0 T10 T290 T30` |
| `G1 X<x> Y<y> Z<z> F<feed>` | Cartesian linear move | `G1 X200 Y80 Z0 F30` |
| `G90` | Absolute positioning mode | |
| `G91` | Incremental positioning mode | |
| `G28` | Home all axes | |
| `G4 P<ms>` | Dwell (pause) | `G4 P500` |

### Gripper Commands

| Command | Description | Example |
|---------|-------------|---------|
| `M3 S<mm> F<feed>` | Close gripper to distance at speed | `M3 S5 F4` |
| `M5 S<mm> F<feed>` | Open gripper to distance at speed | `M5 S20 F4` |
| `M6 F<feed>` | Home gripper | `M6 F4` |

### System Commands

| Command | Description |
|---------|-------------|
| `M17` | Enable stepper motors |
| `M18` | Disable stepper motors |
| `M112` | Emergency stop |
| `M114` | Report current position |
| `M119` | Report endstop states |

### Program Directives

| Directive | Description | Example |
|-----------|-------------|---------|
| `CALL <file>` | Run subroutine file once | `CALL programs/lib/wave.gcode` |
| `CALL <file> N` | Run subroutine file N times | `CALL programs/lib/shapes/circle.gcode 3` |
| `CALL <file> 0` | Run subroutine forever (until Stop) | `CALL programs/lib/wave.gcode 0` |
| `---` | Preamble split marker | Lines above run once, below loop |
| `; text` | Comment (ignored) | `; this is a comment` |

---

## Program Examples

### Basic: Draw a Shape
```gcode
CALL programs/lib/home_safe.gcode
CALL programs/lib/shapes/circle.gcode
G28
```

### Repeat a Shape 5 Times
```gcode
CALL programs/lib/home_safe.gcode
CALL programs/lib/shapes/star.gcode 5
G28
```

### Preamble + Infinite Loop
```gcode
; Home once (preamble)
CALL programs/lib/home_safe.gcode
M17
---
; Loop forever (until Stop)
CALL programs/lib/shapes/cylinder.gcode
G4 P300
```

### Nested Subroutines
```gcode
; demo_nested_call.gcode calls shape_sequence.gcode,
; which itself calls square.gcode, star.gcode, figure8.gcode
CALL programs/lib/home_safe.gcode
CALL programs/lib/shape_sequence.gcode 2
CALL programs/lib/wave.gcode
G28
```

### Pick, Wave, and Place
```gcode
CALL programs/lib/home_safe.gcode
M17
G90
G1 X200 Y30 Z0 F15
CALL programs/lib/gripper_close.gcode
G1 X200 Y120 Z0 F20
CALL programs/lib/wave.gcode
G1 X200 Y30 Z0 F15
CALL programs/lib/gripper_open.gcode
G28
```

---

## Example Programs

Pre-built demo programs are in the `programs/` directory:

| File | Description |
|------|-------------|
| `demo_shapes.gcode` | All basic shapes in sequence (square, triangle, hexagon, circle, star, figure-8) |
| `demo_all_planes.gcode` | Circles in XZ, XY, and YZ planes |
| `demo_spiral.gcode` | Expanding helix screw motion |
| `demo_star_figure8.gcode` | Star + figure-8 + flat spiral |
| `demo_circle_repeat.gcode` | Circle repeated 3× then wave |
| `demo_cylinder_loop.gcode` | Preamble + infinite cylinder loop |
| `demo_infinite_wave.gcode` | Preamble + infinite wave loop |
| `demo_nested_call.gcode` | 2-level nested CALL demonstration |
| `demo_pick_and_wave.gcode` | Gripper + wave + motion combo |

### Subroutine Library

```
programs/lib/
├── gripper_open.gcode      — Open gripper (M5)
├── gripper_close.gcode     — Close gripper (M3)
├── home_safe.gcode         — Home + move to safe hover position
├── wave.gcode              — Side-to-side wave gesture
├── shape_sequence.gcode    — Combo: square → star → figure-8 (for nested CALL demos)
└── shapes/
    ├── square.gcode        — 100mm square in XZ plane
    ├── square_xy.gcode     — 100mm square in XY plane
    ├── triangle.gcode      — Equilateral triangle in XZ plane
    ├── hexagon.gcode       — Hexagon (R=50mm) in XZ plane
    ├── circle.gcode        — Circle (R=50mm, 12-seg) in XZ plane
    ├── circle_xy.gcode     — Circle (R=50mm, 12-seg) in XY plane
    ├── circle_yz.gcode     — Circle (R=50mm, 12-seg) in YZ plane
    ├── star.gcode          — 5-pointed star (R=50mm) in XZ plane
    ├── figure8.gcode       — Figure-8 (two R=25mm loops) in XZ plane
    ├── cylinder.gcode      — Cylinder (R=45mm) between Y=40 and Y=120
    ├── spiral_flat.gcode   — Expanding flat spiral (R=10→50mm) in XZ plane
    └── spiral_helix.gcode  — Expanding helix screw motion (R=15→50mm, Y=30→130)
```

---

## Pick & Place Template Variables

When writing G-code templates in the Vision tab's Pick & Place editor, these variables are resolved at execution time:

| Variable | Source |
|----------|--------|
| `{PICK_X}`, `{PICK_Y}`, `{PICK_Z}` | Pick X/Y/Z spin boxes (or cursor click) |
| `{PLACE_X}`, `{PLACE_Y}`, `{PLACE_Z}` | Selected Place position |
| `{APPROACH_Z}` | Approach Z spin box |
| `{FEED}` | Move Feed spin box |
| `{GRIP_CLOSE}`, `{GRIP_OPEN}` | Gripper close/open distance |
| `{GRIP_FEED}` | Gripper feed speed |

All coordinates in templates are in **world frame** and are automatically converted to robot frame during execution.

---

## Robot Workspace

Based on the 3-DOF arm kinematics (L=140mm, a=54mm):

| Axis | Range | Notes |
|------|-------|-------|
| X | 0 – 320 mm | Radial reach from base |
| Y | ~0 – 180 mm | Height (joint-dependent) |
| Z | -320 – 320 mm | Base rotation sweep |
| θ1 | -90° – 90° | Base rotation |
| θ2 | 0° – 130° | Shoulder |
| θ3 | -17° – 120° | Elbow |

> **Tip:** Stay within X: 140–240, Y: 30–140, Z: ±60 for smooth, step-loss-free operation.

---

## Project Structure

```
Community-Robot-Arm-UI/
├── main.py                          — Application entry point
├── requirements.txt                 — Python dependencies
├── config/
│   └── ui_settings.json             — Default settings (auto-generated)
├── programs/                        — G-code programs and subroutines
│   ├── demo_*.gcode                 — Demo programs
│   └── lib/                         — Reusable subroutine library
│       ├── shapes/                  — Shape subroutines
│       └── *.gcode                  — Utility subroutines
└── robot_arm_ui/
    ├── core/
    │   ├── serial_client.py         — Serial port communication (PyQt signals)
    │   ├── command_executor.py      — Command queue with ack-based flow control
    │   ├── frame_transform.py       — 6-DOF world↔robot coordinate transform
    │   ├── gcode_utils.py           — G-code builder and parser helpers
    │   └── candy_detector.py        — YOLO model wrapper for detection
    ├── models/
    │   └── program_model.py         — Program line storage and file I/O
    └── ui/
        ├── main_window.py           — Main window, sidebar, tabs, signal wiring
        └── vision_widget.py         — Vision tab (camera, calibration, PnP, detection)
```

---

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Escape` | Emergency Stop (M112) — works from any tab |
| `↑` / `↓` | Terminal command history navigation |

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `PyQt6 ≥ 6.7` | GUI framework |
| `pyserial ≥ 3.5` | Serial communication |
| `numpy ≥ 1.24` | Coordinate transforms |
| `opencv-contrib-python ≥ 4.9` | Camera, ArUco detection |
| `ultralytics` *(optional)* | YOLO object detection for auto-sort |

---

## Building Executables

### Local Build (current OS)

```bash
pip install pyinstaller
pyinstaller CommunityRobotArmUI.spec --noconfirm
```

The output will be in `dist/CommunityRobotArmUI/`. Run the executable directly — no Python installation needed on the target machine.

> **Note:** PyInstaller cannot cross-compile. A Windows `.exe` must be built on Windows, a Linux binary on Linux, etc.

### Automated CI/CD (all platforms)

This repo includes a GitHub Actions workflow (`.github/workflows/build.yml`) that automatically builds for **Windows**, **Linux**, and **macOS** when you push a version tag:

```bash
git tag v1.0.0
git push origin v1.0.0
```

This triggers the CI pipeline which:
1. Builds on `windows-latest`, `ubuntu-22.04`, and `macos-13`
2. Packages each as `.zip` (Windows) or `.tar.gz` (Linux/macOS)
3. Creates a GitHub Release with all 3 downloads attached

You can also trigger builds manually from the **Actions** tab in GitHub.

---

## License

See [LICENSE](LICENSE) for details.
