from ultralytics import YOLOv10

if __name__ == '__main__':
    # Load a custom model
    model = YOLOv10('runs/detect/train/weights/best.pt')
    
    # Predict on an image
    results = model.predict(source="ultralytics/assets", device='0', visualize=True, save=True)
    
    # Print results
    print(results)
