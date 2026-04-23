; demo_infinite_wave.gcode — Wave continuously forever
; Preamble homes the robot, then waves in an infinite loop

CALL programs/lib/home_safe.gcode
---
CALL programs/lib/wave.gcode
G4 P500
