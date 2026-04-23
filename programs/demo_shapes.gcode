; demo_shapes.gcode — Draw all basic shapes in sequence
; Demonstrates CALL subroutines for each shape

; Home and safe position
CALL programs/lib/home_safe.gcode

; Square (XZ)
CALL programs/lib/shapes/square.gcode
G4 P400

; Triangle (XZ)
CALL programs/lib/shapes/triangle.gcode
G4 P400

; Hexagon (XZ)
CALL programs/lib/shapes/hexagon.gcode
G4 P400

; Circle (XZ)
CALL programs/lib/shapes/circle.gcode
G4 P400

; Star (XZ)
CALL programs/lib/shapes/star.gcode
G4 P400

; Figure-8 (XZ)
CALL programs/lib/shapes/figure8.gcode

; Return home
G28
