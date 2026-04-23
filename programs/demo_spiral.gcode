; demo_spiral.gcode — Spiral helix demonstration
; Expanding screw motion from tight coil to wide sweep

; Home
CALL programs/lib/home_safe.gcode

; Trace the expanding helix
CALL programs/lib/shapes/spiral_helix.gcode

; Pause at top
G4 P500

; Home
G28
