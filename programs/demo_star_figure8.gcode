; demo_star_figure8.gcode — Star and figure-8 sequence
; Demonstrates complex 2D patterns

; Home
CALL programs/lib/home_safe.gcode

; Draw a star
CALL programs/lib/shapes/star.gcode
G4 P500

; Draw a figure-8
CALL programs/lib/shapes/figure8.gcode
G4 P500

; Flat spiral
CALL programs/lib/shapes/spiral_flat.gcode

; Home
G28
