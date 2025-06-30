from ultralytics import YOLO

# Load a model
model = YOLO("C:\\Users\lenovo\Desktop\helmet-detection\Smart_Construction-master\weights\\best.pt")  # pretrained YOLO11n model

# Run batched inference on a list of images
results = model(["C:\\Users\lenovo\Desktop\helmet-detection\Smart_Construction-master\data\VOC2028\JPEGImages\\000012.jpg"])  # return a list of Results objects

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename="C:\\Users\lenovo\Desktop\helmet-detection\Smart_Construction-master\\result.jpg")  # save to disk