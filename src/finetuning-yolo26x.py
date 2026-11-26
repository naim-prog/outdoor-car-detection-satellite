from ultralytics import YOLO
import datetime


if __name__ == '__main__':

    now_format = datetime.datetime.strftime(datetime.datetime.now(), "%Y-%m-%d_%H-%M-%S")

    model = YOLO(r"model.pt")

    # First finetuning phase
    results = model.train(
        epochs = 60,
        save_period = 3,
        patience = 5,
        batch = 4,
        max_det = 750,
        device = 'cuda:0',
        imgsz = 1024,
        freeze = 10,
        lr0 = 0.001,
        weight_decay = 0.0005,
        cos_lr = True,
        amp = True,
        project = 'fine-tuning_yolo26x-obb',
        name = f'full-dataset_freeze-10_{now_format}',
    )

    # Second finetuning phase
    results = model.train(
        data = r"X:\datasets\outdoor-car-detection-satellite\data.yaml",
        device = 'cuda:0',
        epochs = 40,
        patience = 10,
        imgsz = 1024,
        batch = 4,
        optimizer = "AdamW",
        lr0 = 0.0003,
        cos_lr = True,
        warmup_epochs = 1,
        weight_decay = 0.0005,
        freeze = 0,
        max_det = 750,
        amp = True,
        save_period = 2,
        project = 'fine-tuning_yolo26x-obb',
        name = f'full-dataset_unfreeze_40-epochs',
    )
