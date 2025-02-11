from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from collections import deque

app = Flask(__name__)
app.secret_key = 'some_secret_key'

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


class Parameters:
    def __init__(self):
        self.CLASSES = self.load_classes("model/action_recognition_kinetics.txt")
        self.ACTION_RESNET = 'model/resnet-34_kinetics.onnx'
        self.SAMPLE_DURATION = 16
        self.SAMPLE_SIZE = 112
        self.CONFIDENCE_THRESHOLD = 0.5
        self.PLAYBACK_SPEED = 2

    def load_classes(self, class_file):
        with open(class_file, 'r') as f:
            classes = f.read().strip().split("\n")
        custom_classes = ["jumping", "sliding", "playing in indoor playground"]
        classes.extend(custom_classes)
        return classes

class ActionRecognition:
    def __init__(self, params):
        self.params = params
        self.net = cv2.dnn.readNet(model=params.ACTION_RESNET)
        self.captures = deque(maxlen=params.SAMPLE_DURATION)

    def preprocess_frame(self, frame):
        return cv2.resize(frame, dsize=(550, 400))

    def predict(self):
        imageBlob = cv2.dnn.blobFromImages(self.captures, 1.0,
                                           (self.params.SAMPLE_SIZE, self.params.SAMPLE_SIZE),
                                           (114.7748, 107.7354, 99.4750),
                                           swapRB=True, crop=True)
        imageBlob = np.transpose(imageBlob, (1, 0, 2, 3))
        imageBlob = np.expand_dims(imageBlob, axis=0)
        self.net.setInput(imageBlob)
        outputs = self.net.forward()
        return outputs[0]

    def get_best_prediction(self, outputs):
        best_index = np.argmax(outputs)
        return self.params.CLASSES[best_index], outputs[best_index]



def scale_confidence(confidence):
    """Scale the confidence to be between 0 and 100."""
    return min(confidence * 100, 100)        

def process_video(video_path):
    params = Parameters()
    params.VIDEO_PATH = video_path
    action_recognition = ActionRecognition(params)
    cap = cv2.VideoCapture(params.VIDEO_PATH)
    cap.set(cv2.CAP_PROP_AUDIO_STREAM, -1)  # Disable audio

    all_predictions = []

    while True:
        for _ in range(params.PLAYBACK_SPEED):
            ret, frame = cap.read()
            if not ret:
                break

        if not ret:
            break

        processed_frame = action_recognition.preprocess_frame(frame)
        action_recognition.captures.append(processed_frame)

        if len(action_recognition.captures) == params.SAMPLE_DURATION:
            outputs = action_recognition.predict()
            prediction = action_recognition.get_best_prediction(outputs)
            all_predictions.append(prediction)

    cap.release()

    best_prediction = max(all_predictions, key=lambda x: x[1])
    scaled_confidence = scale_confidence(best_prediction[1])
    return (best_prediction[0], scaled_confidence), os.path.basename(video_path)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            best_prediction, video_filename = process_video(filepath)
            return render_template('result.html', prediction=best_prediction, video_filename=video_filename)
    return render_template('upload.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)