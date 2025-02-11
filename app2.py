from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
import os
from werkzeug.utils import secure_filename
import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import numpy as np
from collections import deque
from torch.nn import Transformer
import threading
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)
app.secret_key = 'some_secret_key'

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global model instance to avoid reloading
global_model = None
global_params = None

class Parameters:
    def __init__(self):
        self.CLASSES = self.load_classes("model/action_recognition_kinetics.txt")
        self.SAMPLE_DURATION = 16
        self.SAMPLE_SIZE = 112
        self.CONFIDENCE_THRESHOLD = 0.5
        self.PLAYBACK_SPEED = 4  # Increased playback speed
        self.BATCH_SIZE = 4  # Added batch processing

    def load_classes(self, class_file):
        try:
            with open(class_file, 'r') as f:
                classes = f.read().strip().split("\n")
            custom_classes = ["jumping", "sliding", "playing in indoor playground"]
            classes.extend(custom_classes)
            return classes
        except Exception as e:
            flash(f"Error loading classes: {e}")
            return []

class AttentionModel(nn.Module):
    def __init__(self, num_classes, sample_duration):
        super(AttentionModel, self).__init__()
        self.resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        self.resnet.fc = nn.Identity()
        # Simplified transformer for faster processing
        self.transformer = Transformer(d_model=512, nhead=4, num_encoder_layers=2)
        self.fc = nn.Linear(512, num_classes)

    @torch.inference_mode()  # Faster than no_grad()
    def forward(self, x):
        b, t, c, h, w = x.size()
        x = x.view(b * t, c, h, w)
        features = self.resnet(x)
        features = features.view(b, t, 512)
        features = self.transformer(features, features)
        features = torch.mean(features, dim=1)
        return self.fc(features)

class ActionRecognition:
    def __init__(self, params):
        self.params = params
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        global global_model
        if global_model is None:
            global_model = AttentionModel(num_classes=len(params.CLASSES), 
                                        sample_duration=params.SAMPLE_DURATION).to(self.device)
            global_model.eval()
        self.model = global_model
        self.captures = []

    def preprocess_frames_batch(self, frames):
        """Process multiple frames at once"""
        processed_frames = []
        for frame in frames:
            frame = cv2.resize(frame, (self.params.SAMPLE_SIZE, self.params.SAMPLE_SIZE))
            frame = frame.transpose(2, 0, 1)
            frame = frame / 255.0
            processed_frames.append(frame)
        return np.array(processed_frames)

    @torch.inference_mode()
    def predict_batch(self, frames_batch):
        """Predict on a batch of frames"""
        frames = torch.tensor(frames_batch, dtype=torch.float32).to(self.device)
        outputs = self.model(frames)
        return outputs

    def get_best_prediction(self, outputs):
        _, best_index = torch.max(outputs, dim=1)
        confidence = torch.softmax(outputs, dim=1)[0, best_index].item()
        return self.params.CLASSES[best_index.item()], confidence

def extract_frames(video_path, sample_rate):
    """Extract frames efficiently"""
    frames = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame indices to extract
    frame_indices = np.linspace(0, total_frames-1, 32, dtype=int)
    
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    
    cap.release()
    return frames

def process_video(video_path):
    try:
        global global_params
        if global_params is None:
            global_params = Parameters()
        params = global_params
        
        action_recognition = ActionRecognition(params)
        
        # Extract frames efficiently
        frames = extract_frames(video_path, params.PLAYBACK_SPEED)
        if not frames:
            flash("No frames could be extracted from the video")
            return None, None

        # Process frames in batches
        all_predictions = []
        batch_size = params.BATCH_SIZE
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i + batch_size]
            if len(batch_frames) == batch_size:
                processed_batch = action_recognition.preprocess_frames_batch(batch_frames)
                batch_tensor = torch.tensor(processed_batch, dtype=torch.float32).unsqueeze(0)
                outputs = action_recognition.predict_batch(batch_tensor)
                prediction = action_recognition.get_best_prediction(outputs)
                all_predictions.append(prediction)

        if not all_predictions:
            flash("No predictions were made.")
            return None, None

        best_prediction = max(all_predictions, key=lambda x: x[1])
        scaled_confidence = min(best_prediction[1] * 100, 100)
        return (best_prediction[0], scaled_confidence), os.path.basename(video_path)
    
    except Exception as e:
        flash(f"Error processing video: {e}")
        return None, None

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
            try:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                prediction_result = process_video(filepath)
                if prediction_result is None:
                    return redirect(request.url)
                    
                best_prediction, video_filename = prediction_result
                return render_template('result.html', 
                                     prediction=best_prediction, 
                                     video_filename=video_filename, 
                                     min=min)
            except Exception as e:
                flash(f'Error processing file: {e}')
                return redirect(request.url)
                
    return render_template('upload.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    # Initialize model at startup
    params = Parameters()
    action_recognition = ActionRecognition(params)
    global_params = params
    
    # Run Flask app
    app.run(debug=True, threaded=True)