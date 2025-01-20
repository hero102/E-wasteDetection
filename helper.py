from ultralytics import YOLO
import pickle


with open('D:\FinelYearProject\e-waste\streamlit-detection-tracking - app\weights\yolov8 (1).pkl', 'rb') as file:
    model1= pickle.load(file)



def load_model(model_path):
    model = YOLO('D:\FinelYearProject\e-waste\streamlit-detection-tracking - app\weights/yolov8n.pt')
    return model


 

