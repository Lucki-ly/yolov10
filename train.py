from ultralytics import YOLOv10

if __name__ == '__main__':
    model = YOLOv10(model="H:/xiangmu/2/yolov10/ultralytics/cfg/models/v10/yolov10n.yaml").load('yolov10n.pt')  # 从头开始构建新模型
    #model = YOLOv10.from_pretrained('jameslahm/yolov10n')
    #print(model)
    
    # 使用模型
    results = model.train(
        data="coco.yaml",
        patience=0,
        epochs=50,
        device='0',
        batch=16,
        seed=42
    )  # 训练模型
