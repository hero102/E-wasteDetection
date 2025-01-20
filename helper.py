from ultralytics import YOLO

def load_model(model_path):
    model = YOLO('D:\FinelYearProject\e-waste\streamlit-detection-tracking - app\weights/yolov8n.pt')
    return model


 

