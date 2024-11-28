from flask import Flask, render_template, request, jsonify
from playsound import playsound
from twilio.rest import Client
import threading
import os
import time
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import numpy as np
import re

app = Flask(__name__)

# Twilio Configuration
TWILIO_ACCOUNT_SID = 'ACf986c62ffae4902bbc5771b2c6906a9c'
TWILIO_AUTH_TOKEN = '4c91e709828579a556cc2c73d810193e'
TWILIO_PHONE_NUMBER = '+16814122086'
RECIPIENT_PHONE_NUMBER = '+918660486877'

# Load YAMNet Model
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
class_map_path = yamnet_model.class_map_path().numpy().decode('utf-8')
class_labels = [line.strip().split(",")[2].strip('"') for line in open(class_map_path).readlines()]

# Unsafe sound categories
unsafe_classes = [
    "Air horn", "cry", "Explosion", "chuckle", "smashing", "Snort", 
    "shout", "scream", "Gunshot", "drilling", "Glass break", "crack"
]

# Global flags and variables
alert_active = False
monitoring = True  # Tracks the monitoring state
latest_alert = {}  # Stores the latest alert details
processed_files = set()  # Track processed files globally
alert_logs = []  # Global list to store alert logs

# --- Helper Functions ---
def preprocess_label(label):
    label = label.lower()
    label = re.sub(r'(ing|n)$', '', label)
    return label

# Function to load and preprocess audio
def load_audio(file_path):
    waveform, sample_rate = librosa.load(file_path, sr=16000)
    return waveform


# Function to classify threat level
def classify_threat_level(confidence):
    if confidence >= 0.7:
        return "High"
    elif confidence >= 0.4:
        return "Medium"
    else:
        return "Low"

# Function to classify sound
def classify_sound(file_path):
    global alert_active
    waveform = load_audio(file_path)
    scores, embeddings, spectrogram = yamnet_model(waveform)
    prediction = scores.numpy().mean(axis=0)
    top_class_index = np.argmax(prediction)
    top_class_label = class_labels[top_class_index]
    confidence = prediction[top_class_index]
    processed_label = preprocess_label(top_class_label)

    for unsafe_class in unsafe_classes:
        if preprocess_label(unsafe_class) in processed_label:
            threat_level = classify_threat_level(confidence)
            alert_active = True
            trigger_alerts(file_path, top_class_label, confidence, threat_level)
            break


def play_audio_alert():
    playsound('audioset/danger/cry1.wav') 

# Function to send SMS notification
def send_sms_notification():
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    client.messages.create(
        body="Unsafe sound detected! Please check your system immediately.",
        from_=TWILIO_PHONE_NUMBER,
        to=RECIPIENT_PHONE_NUMBER
    )

# Function to make a call
def make_call():
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    client.calls.create(
        twiml='<Response><Say>Alert! Unsafe sound detected at your location. Please respond immediately.</Say></Response>',
        from_=TWILIO_PHONE_NUMBER,
        to=RECIPIENT_PHONE_NUMBER
    )

def trigger_alerts(file_path, sound_label, confidence, threat_level):
    global alert_active, latest_alert, alert_logs

    # Update alert details
    alert_active = True
    latest_alert = {
        "sound_label": sound_label,
        "confidence": float(confidence),
        "threat_level": threat_level
    }

    # Add log entry
    log_entry = {
        "sound_label": sound_label,
        "confidence": f"{confidence:.2f}",
        "threat_level": threat_level,
        "message": f"Alert: {sound_label} detected with confidence {confidence:.2f}. Threat Level: {threat_level}"
    }
    alert_logs.append(log_entry)
    print(log_entry["message"])  # Print log in the console

    # Conditional actions based on threat level
    if threat_level == "Low":
        threading.Thread(target=send_sms_notification).start()
    elif threat_level == "High":
        threading.Thread(target=play_audio_alert).start()
        threading.Thread(target=send_sms_notification).start()
        threading.Thread(target=make_call).start()
    elif threat_level == "Medium":
        threading.Thread(target=send_sms_notification).start()
        threading.Thread(target=make_call).start()


def process_audio_folder(folder_path):
    global monitoring, processed_files
    while monitoring:
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.wav') and file_name not in processed_files:
                file_path = os.path.join(folder_path, file_name)
                processed_files.add(file_name)  # Add before classification
                classify_sound(file_path)
        time.sleep(1)

# --- Flask Routes ---

@app.route('/')
def home():
    global alert_active, latest_alert
    alert_active = False
    latest_alert = {}
    return render_template('index.html')

@app.route('/status', methods=['GET'])
def status():
    global alert_active, latest_alert
    return jsonify({
        "monitoring": monitoring,
        "alert": latest_alert if alert_active else None
    })


@app.route('/dismiss_alert', methods=['POST'])
def dismiss_alert():
    global alert_active, latest_alert
    alert_active = False
    latest_alert = {}
    return jsonify({"status": "dismissed", "message": "Alerts have been dismissed."})

@app.route('/start_monitoring', methods=['POST'])
def start_monitoring():
    global monitoring
    if not monitoring:
        monitoring = True
        threading.Thread(target=process_audio_folder, args=('audioset/unsafe',), daemon=True).start()
        return jsonify({"status": "started", "message": "Monitoring started."})
    return jsonify({"status": "already_started", "message": "Monitoring is already active."})


@app.route('/logs', methods=['GET'])
def get_logs():
    """API to fetch alert logs"""
    return jsonify({"logs": alert_logs})


@app.route('/stop_monitoring', methods=['POST'])
def stop_monitoring():
    global monitoring
    if monitoring:
        monitoring = False
        return jsonify({"status": "stopped", "message": "Monitoring stopped."})
    return jsonify({"status": "already_stopped", "message": "Monitoring is not active."})

# --- Main ---
if __name__ == '__main__':
    folder_path = 'audioset/unsafe'
    audio_thread = threading.Thread(target=process_audio_folder, args=(folder_path,), daemon=True)
    audio_thread.start()
    app.run(debug=True)
