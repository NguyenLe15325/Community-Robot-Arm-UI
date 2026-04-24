"""Debug script to inspect ONNX model output shape and metadata."""
import sys
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    print("pip install onnxruntime")
    sys.exit(1)

model_path = sys.argv[1] if len(sys.argv) > 1 else input("ONNX model path: ")

session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

# Metadata
meta = session.get_modelmeta().custom_metadata_map
print("=== METADATA ===")
for k, v in meta.items():
    print(f"  {k}: {v[:200] if len(v) > 200 else v}")

# Inputs
print("\n=== INPUTS ===")
for inp in session.get_inputs():
    print(f"  {inp.name}: shape={inp.shape}, dtype={inp.type}")

# Outputs
print("\n=== OUTPUTS ===")
for out in session.get_outputs():
    print(f"  {out.name}: shape={out.shape}, dtype={out.type}")

# Run with dummy image
input_info = session.get_inputs()[0]
shape = input_info.shape  # e.g. [1, 3, 640, 640]
h, w = int(shape[2]), int(shape[3])
dummy = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

import cv2
rgb = cv2.cvtColor(dummy, cv2.COLOR_BGR2RGB)
blob = rgb.astype(np.float32) / 255.0
blob = blob.transpose(2, 0, 1)[np.newaxis, ...]

outputs = session.run(None, {input_info.name: blob})

print(f"\n=== RUNTIME OUTPUT ===")
for i, out in enumerate(outputs):
    arr = np.array(out)
    print(f"  output[{i}]: shape={arr.shape}, dtype={arr.dtype}, min={arr.min():.4f}, max={arr.max():.4f}")

# Parse class names
import ast
names_str = meta.get("names", "")
if names_str:
    names_dict = ast.literal_eval(names_str)
    n_cls = len(names_dict)
    print(f"\n=== CLASSES ({n_cls}) ===")
    for k, v in names_dict.items():
        print(f"  {k}: {v}")
    
    # Check output[0] layout
    preds = outputs[0][0]  # (features, boxes)
    print(f"\n=== ANALYSIS ===")
    print(f"  Raw output[0][0] shape: {preds.shape}")
    print(f"  Expected: (4 + {n_cls} = {4+n_cls}, num_boxes) for detection")
    print(f"  Expected: (4 + {n_cls} + 32 = {4+n_cls+32}, num_boxes) for segmentation")
    print(f"  Actual feature dim: {preds.shape[0]}")
    
    if preds.shape[0] == 4 + n_cls:
        print(f"  → This is a DETECTION model")
    elif preds.shape[0] == 4 + n_cls + 32:
        print(f"  → This is a SEGMENTATION model (32 mask coefficients)")
    else:
        print(f"  → UNKNOWN layout! Diff from detection: {preds.shape[0] - 4 - n_cls}")
