; demo_nested_call.gcode — Demonstrates nested CALL subroutines
; This program calls shape_sequence.gcode, which itself calls
; individual shape subroutines — showing nested CALL in action

; Home first
CALL programs/lib/home_safe.gcode

; Call the combo subroutine (which internally calls shapes)
CALL programs/lib/shape_sequence.gcode 2

; Wave goodbye
CALL programs/lib/wave.gcode

; Home
G28
M18
