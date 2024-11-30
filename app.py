from flask import Flask, request, jsonify, render_template
import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import mediapipe as mp
import joblib


app = Flask(__name__)

model = tf.keras.models.load_model("gait_classification_model.h5")

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Extract features
def extract_pose_features(video_path):
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(video_path)
    pose_features = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            frame_features = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks]).flatten()
            pose_features.append(frame_features)
        else:
            pose_features.append(np.zeros(132))  

    cap.release()
    pose.close()

    return np.mean(pose_features, axis=0)

scaler = joblib.load("scaler.pkl")

# classify video
def classify_video(video_path):
    features = extract_pose_features(video_path)

    features = features.reshape(1, -1)  
    features = scaler.transform(features)  

    features = features.reshape((features.shape[0], 1, features.shape[1]))

    prediction = model.predict(features)
    return "Normal Gait" if prediction < 0.5 else "Abnormal Gait"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video = request.files['video']
    video_path = os.path.join("uploads", video.filename)

    os.makedirs("uploads", exist_ok=True)
    video.save(video_path)

    try:
        result = classify_video(video_path)
        return jsonify({"result": result}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)

if __name__ == '__main__':
    app.run(debug=True)