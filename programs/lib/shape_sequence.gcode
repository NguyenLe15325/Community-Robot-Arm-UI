; shape_sequence.gcode — A combo subroutine called by demo_nested_call
; Itself calls shape subroutines (nested CALL)

; Square
CALL programs/lib/shapes/square.gcode
G4 P300

; Star
CALL programs/lib/shapes/star.gcode
G4 P300

; Figure-8
CALL programs/lib/shapes/figure8.gcode
G4 P300

; Return to center
G90
G1 X190 Y100 Z0 F25
