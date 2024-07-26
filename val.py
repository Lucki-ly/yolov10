from ultralytics import YOLOv10

if __name__ == '__main__':
    # Load a custom model
    model = YOLOv10('runs/detect/train/weights/best.pt')
    
    # Validate the model
    metrics = model.val(split='val', save_json=True)
