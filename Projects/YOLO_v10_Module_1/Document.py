from pathlib import Path
from ultralytics import YOLO

if __name__ == "__main__":
    current_dir = str(
        Path(__file__).parent if "__file__" in locals() else Path.cwd())
    MODEL_PATH = current_dir + "\\yolov10n.pt"
    model = YOLO(MODEL_PATH)
    YAML_PATH = current_dir + "\\datasets\\safety_helmet_dataset\\data.yaml"
    IMG_SIZE = 640
    EPOCHS = 50
    BATCH_SIZE = 16

    model.train(data=YAML_PATH, imgsz=IMG_SIZE,
                epochs=EPOCHS, batch=BATCH_SIZE)
    TRAINED_MODEL_PATH = current_dir + \
        "\\yolov10\\runs\\detect\\train\\weights\\best.pt"
    model = YOLO(TRAINED_MODEL_PATH)

    model.val(data=YAML_PATH, imgsz=IMG_SIZE, split='test')
