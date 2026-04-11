# Community Robot Arm UI (PyQt)

Desktop UI for the Community Robot Arm firmware.

## Features

- Serial connection manager (port, baud, connect/disconnect)
- ACK-driven command queue (`next command` sent only after `ok`)
- Cartesian control in robot frame or world frame
- Configurable world frame transform (`robot -> world`)
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

The transform is defined as:

- world = R(robot) + t
- `R = Rz(yaw) * Ry(pitch) * Rx(roll)`

For world-frame command input:

- absolute move: world position -> robot position via inverse transform
- relative/jog move: world delta -> robot delta via inverse rotation (no translation)

## Program Files

Program files are plain text (`.gcode`, `.nc`, `.txt`), one command per line.

- empty lines are ignored
- lines starting with `;` are comments and ignored at execution
