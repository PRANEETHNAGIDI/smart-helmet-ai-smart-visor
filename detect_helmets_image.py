from ultralytics import YOLO
import cv2
import os
from datetime import datetime

# =========================
# CONFIG
# =========================

MODEL_PATH = "my_helmet_model_2class.pt"  # your trained model
IMAGE_PATH = "test4.jpg"                  # input image
VIOLATIONS_DIR = "violations"             # folder to save "Without Helmet" crops
OUTPUT_DIR = "outputs"                    # folder to save full annotated images
MAX_DISPLAY_WIDTH = 1000                  # for display scaling
CONF_THRESHOLD = 0.18                     # tweak 0.15–0.25

# =========================
# LOAD MODEL
# =========================

model = YOLO(MODEL_PATH)

# =========================
# LOAD IMAGE
# =========================

img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"Could not read image: {IMAGE_PATH}")

# Ensure directories exist
os.makedirs(VIOLATIONS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Base name for saving files
base_name = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# =========================
# RUN PREDICTION
# =========================

results = model(img, conf=CONF_THRESHOLD)

class_names = model.names
print("DEBUG CLASSES:", class_names)  # should be {0: 'With Helmet', 1: 'Without Helmet'}

violation_count = 0
output = img.copy()

def normalize(name: str) -> str:
    """Lowercase and remove spaces for easy comparison."""
    return name.lower().replace(" ", "")

# =========================
# PROCESS DETECTIONS
# =========================

h_img, w_img = output.shape[:2]

# Bigger, adaptive font size
font_scale = max(0.8, min(1.5, w_img / 800))  # auto-adjust with width
font_thickness = 2

for r in results:
    for box in r.boxes:
        cls_id = int(box.cls[0])
        cls_name = class_names[cls_id]   # "With Helmet" or "Without Helmet"
        norm = normalize(cls_name)
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])

        # Convert to int
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        print(f"Detected {cls_name} with confidence {conf:.2f} at [{x1}, {y1}, {x2}, {y2}]")

        # Clamp to image bounds
        x1 = max(0, min(x1, w_img - 1))
        x2 = max(0, min(x2, w_img - 1))
        y1 = max(0, min(y1, h_img - 1))
        y2 = max(0, min(y2, h_img - 1))

        if x2 <= x1 or y2 <= y1:
            print(f"WARNING: Invalid box after clamping, skipping crop: [{x1}, {y1}, {x2}, {y2}]")
            continue

        if norm == "withouthelmet":
            # VIOLATION
            color = (0, 0, 255)  # red
            label = f"Without Helmet ({conf:.2f})"

            # Make crop
            crop = output[y1:y2, x1:x2]

            if crop.size == 0:
                print(f"WARNING: Crop is empty, not saving. Box: [{x1}, {y1}, {x2}, {y2}]")
            else:
                violation_count += 1
                crop_name = f"{base_name}_violation_{timestamp}_{violation_count}.jpg"
                save_path = os.path.join(VIOLATIONS_DIR, crop_name)
                success = cv2.imwrite(save_path, crop)
                print(f"Saved violation crop -> {save_path} | Success: {success} | Crop shape: {crop.shape}")

        elif norm == "withhelmet":
            # SAFE
            color = (0, 255, 0)  # green
            label = f"With Helmet ({conf:.2f})"
        else:
            # Any unexpected class
            color = (255, 255, 255)
            label = f"{cls_name} ({conf:.2f})"

        # ======= Draw bounding box =======
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

        # ======= Draw filled box behind text (for readability) =======
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        # Text background rectangle
        cv2.rectangle(
            output,
            (x1, y1 - text_h - 10),
            (x1 + text_w + 4, y1),
            (0, 0, 0),
            thickness=-1,
        )
        # Text itself
        cv2.putText(
            output,
            label,
            (x1 + 2, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),  # white text
            font_thickness,
        )

# =========================
# SAVE FULL ANNOTATED IMAGE
# =========================

annotated_name = f"{base_name}_annotated_{timestamp}.jpg"
annotated_path = os.path.join(OUTPUT_DIR, annotated_name)
cv2.imwrite(annotated_path, output)
print(f"Saved full annotated image -> {annotated_path}")

# =========================
# RESIZE FOR DISPLAY
# =========================

display = output.copy()
h, w = display.shape[:2]

if w > MAX_DISPLAY_WIDTH:
    scale = MAX_DISPLAY_WIDTH / w
    new_w = int(w * scale)
    new_h = int(h * scale)
    display = cv2.resize(display, (new_w, new_h))

cv2.namedWindow("Helmet Detection", cv2.WINDOW_NORMAL)
cv2.imshow("Helmet Detection", display)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"Total 'Without Helmet' violations: {violation_count}")
print(f"Check '{VIOLATIONS_DIR}' for crops and '{OUTPUT_DIR}' for annotated images.")
