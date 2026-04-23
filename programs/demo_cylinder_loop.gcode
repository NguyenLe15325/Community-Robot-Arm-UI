; demo_cylinder_loop.gcode — Continuously trace cylinders
; Uses preamble (---) to home once, then loops the cylinder forever

; Preamble: runs once
CALL programs/lib/home_safe.gcode
M17
---
; Loop body: runs forever (until Stop)
CALL programs/lib/shapes/cylinder.gcode
G4 P300
