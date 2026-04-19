# Community Robot Arm UI (PyQt)

Desktop UI for the Community Robot Arm firmware.

## Features

- Serial connection manager (port, baud, connect/disconnect)
- ACK-driven command queue (`next command` sent only after `ok`)
- Cartesian control in robot frame or world frame
- Configurable world frame transform (`world -> robot`)
- Jog controls (`+/-X`, `+/-Y`, `+/-Z`)
- Joint move controls (`T1/T2/T3`)
- Integrated terminal log + manual command send
- Program editor with record/save/load/execute
- Emergency stop (`M112`) and queue clearing

## Protocol Notes

This UI is aligned with the firmware protocol in `Community-Robot-Arm`:

- commands are line-based G-code
- success ACK is `ok`
- errors start with `Error:`
- emergency alarm starts with `ALARM:`

The executor only dispatches a queued command when the previous command has finished (`ok` received).

## Install

```powershell
cd Community-Robot-Arm-UI
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```powershell
cd Community-Robot-Arm-UI
python main.py
```

## World Frame Transform

The transform converts world coordinates to robot coordinates using a **translation-then-rotation** sequence:

**Formula:** `p_robot = R(roll, pitch, yaw) × (p_world − t)`

**Conversion steps:**
1. Subtract translation: `(p_world - t) = (p_world - tx, p_world - ty, p_world - tz)`
2. Apply rotation matrix: `R = Rz(yaw) × Ry(pitch) × Rx(roll)`
   - Matrix is built in ZYX order
   - When applied to points, effective rotation sequence is X → Y → Z (roll → pitch → yaw)
3. Result is robot coordinates sent to robot arm

**Parameters to calibrate:**
- `tx, ty, tz`: Position of robot origin in world coordinates (mm)
- `roll, pitch, yaw`: Angles that map world axes to robot axes (degrees)

**Examples:**
- If robot origin is at world (100, 254, 105), set `tx=100, ty=254, tz=105`
- If world frame is rotated 45° about Z relative to robot, set `yaw=-45`

For world-frame command input in the UI:
- **Absolute move**: world position → applies full transform → robot position
- **Relative/jog move**: world delta → applies rotation only (no translation)

## Program Files

Program files are plain text (`.gcode`, `.nc`, `.txt`), one command per line.

- empty lines are ignored
- lines starting with `;` are comments and ignored at execution
