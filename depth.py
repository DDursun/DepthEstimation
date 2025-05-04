import torch
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
import depth_pro
from ultralytics import YOLO
import time
import os

# Load the depth model
model, transform = depth_pro.create_model_and_transforms()
model.eval()

# Load YOLO model (pretrained)
yolo_model = YOLO("yolo11n.pt")  # Using YOLOv11 nano model

# Load and preprocess the image for YOLO
image_path = "data/cup.jpg"
image_pil = Image.open(image_path).convert("RGB")
image_rgb = np.array(image_pil)
image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

# YOLO model
_ = yolo_model(image_rgb)
start_time = time.time()
results = yolo_model(image_rgb)
end_time = time.time()
yolo_inference_time = end_time - start_time
print(f"YOLO Inference Time: {yolo_inference_time:.4f} seconds")

# Extract bounding boxes, class labels, and confidences
boxes = []
for r in results:
    boxes.extend(r.boxes.data.cpu().numpy())  
labels = yolo_model.names 

# depth estimation
image, _, f_px = depth_pro.load_rgb(image_path)
image_tensor = transform(image).unsqueeze(0)
with torch.no_grad():
    _ = model.infer(image_tensor, f_px=f_px)

start_time = time.time()
with torch.no_grad():
    prediction = model.infer(image_tensor, f_px=f_px)
end_time = time.time()
depth_inference_time = end_time - start_time
print(f"Depth Inference Time: {depth_inference_time:.4f} seconds")

# Retrieve depth map (depth in meters)
depth = prediction["depth"].squeeze().cpu().numpy()

# Iterate over detected objects
for box in boxes:
    xmin, ymin, xmax, ymax, confidence, class_id = box
    label = labels[int(class_id)]

    # Calculate the midpoint of the bounding box
    midpoint_x = int((xmin + xmax) / 2)
    midpoint_y = int((ymin + ymax) / 2)

    # Ensure the midpoint is within the depth map bounds
    h, w = depth.shape
    midpoint_x = np.clip(midpoint_x, 0, w - 1)
    midpoint_y = np.clip(midpoint_y, 0, h - 1)

    # Retrieve the depth at the midpoint from the depth map
    object_depth = depth[midpoint_y, midpoint_x]  

    # Print the object name and its estimated distance
    print(f"Object: {label}, Estimated Distance: {object_depth:.2f} meters")

    # Draw the bounding box and label on the image
    cv2.rectangle(image_bgr, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
    cv2.putText(image_bgr, f"{label} ({object_depth:.2f}m)", (int(xmin), int(ymin) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Display the result
plt.imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title("Detected Objects with Estimated Distance")
plt.show()

# Save
output_path = os.path.join(os.path.dirname(image_path), 'annotated_output_cups.jpg')
cv2.imwrite(output_path, image_bgr)
print(f"Annotated image saved to: {output_path}")
