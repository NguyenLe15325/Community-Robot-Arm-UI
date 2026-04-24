G90                                                     ; absolute mode
G1 X{PICK_X} Y{PICK_Y} Z{APPROACH_Z} F{MOVE_FEED}      ; approach above pick
M6                                                      ; gripper home
G1 X{PICK_X} Y{PICK_Y} Z{PICK_Z} F{MOVE_FEED}          ; descend to pick
M3 S{GRIP_CLOSE} F{GRIP_FEED}                           ; gripper close
G1 X{PICK_X} Y{PICK_Y} Z{APPROACH_Z} F{PICK_FEED}      ; retract up from pick
G1 X{PLACE_X} Y{PLACE_Y} Z{APPROACH_Z} F{PICK_FEED}    ; approach above place
G1 X{PLACE_X} Y{PLACE_Y} Z{PLACE_Z} F{PICK_FEED}       ; descend to place
M5 S{GRIP_OPEN} F{GRIP_FEED}                            ; gripper open
G1 X{PLACE_X} Y{PLACE_Y} Z{APPROACH_Z} F{MOVE_FEED}    ; retract up from place