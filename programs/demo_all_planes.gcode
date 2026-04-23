; demo_all_planes.gcode — Draw circles in all 3 planes
; Shows the robot's full 3D range

; Home
CALL programs/lib/home_safe.gcode

; Circle in XZ plane (top-down)
CALL programs/lib/shapes/circle.gcode
G4 P500

; Circle in XY plane (side view)
CALL programs/lib/shapes/circle_xy.gcode
G4 P500

; Circle in YZ plane (front view)
CALL programs/lib/shapes/circle_yz.gcode
G4 P500

; Home
G28
