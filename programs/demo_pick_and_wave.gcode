; demo_pick_and_wave.gcode — Grip an object, wave, then release
; Demonstrates gripper subroutines mixed with motion

; Home and enable
CALL programs/lib/home_safe.gcode
M17

; Move above object
G90
G1 X200 Y80 Z0 F25

; Descend and grab
G1 X200 Y30 Z0 F15
CALL programs/lib/gripper_close.gcode
G4 P500

; Lift
G1 X200 Y120 Z0 F20

; Wave while holding
CALL programs/lib/wave.gcode

; Put it back
G1 X200 Y30 Z0 F15
CALL programs/lib/gripper_open.gcode
G4 P300

; Retract
G1 X200 Y120 Z0 F25

; Home
G28
