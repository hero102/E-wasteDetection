from flask import Flask, render_template, Response, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
from pathlib import Path
import cv2
import settings
import helper
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)
app.secret_key = "supersecretkey"

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the pre-trained model
try:
    model = helper.load_model(settings.DETECTION_MODEL)
except Exception as ex:
    model = None
    print(f"Error loading model: {ex}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Perform object detection
        try:
            uploaded_image = Image.open(file_path)
            res = model.predict(uploaded_image, conf=0.4)  # Using default confidence
            res_plotted = res[0].plot()[:, :, ::-1]  # Convert to RGB for saving
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], f"detected_{filename}")
            Image.fromarray(res_plotted).save(output_path)

            return render_template('result.html',
                                   uploaded_file=url_for('static', filename=f'uploads/{filename}'),
                                   detected_file=url_for('static', filename=f'uploads/detected_{filename}'),
                                   boxes=res[0].boxes)

        except Exception as ex:
            flash(f"Error processing file: {ex}")
            return redirect(url_for('index'))

@app.route('/webcam')
def webcam():
    return render_template('webcam.html')

def generate_frames():
    cap = cv2.VideoCapture(0)  # Open webcam (default camera)

    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            try:
                # Perform object detection on the current frame
                results = model.predict(frame, conf=0.4)
                res_plotted = results[0].plot()  # Overlay detections on the frame
                _, buffer = cv2.imencode('.jpg', res_plotted)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except Exception as ex:
                print(f"Error during webcam detection: {ex}")
                break

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
