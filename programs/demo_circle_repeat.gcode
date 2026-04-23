; demo_circle_repeat.gcode — Trace circles 3 times then wave
; Demonstrates CALL with repeat count

; Home first
CALL programs/lib/home_safe.gcode

; Trace circle 3 times
CALL programs/lib/shapes/circle.gcode 3

; Celebrate with a wave
CALL programs/lib/wave.gcode

; Home
G28
