from ultralytics import YOLO


class model_training:

    def __init__(self) -> None:
        pass 

    def train_model(self):

        model = YOLO("yolov8n.pt")  # build a new model from scratch
        # Use the model
        model.train(data="./data.yaml", epochs=150,batch=-1,patience=20)
