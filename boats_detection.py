from ultralytics import YOLO

# Create a new YOLO model from scratch
model = YOLO()

# Load a pretrained YOLO model (recommended for training)
#model = YOLO('yolov8n.pt')

# Train the model using the '/content/data.yaml' dataset for 3 epochs
results = model.train(data='/content/data.yaml', epochs=10)

# Perform object detection on an image using the model
# results = model('https://ultralytics.com/images/bus.jpg')

# Export the model to ONNX format
# success = model.export(format='onnx')

# Evaluate the model's performance on the validation set
results = model.val()

results = model('/content/train_data/images/validation/1667846701.393603.jpg')

# Export the model to ONNX format
success = model.export(format='onnx')