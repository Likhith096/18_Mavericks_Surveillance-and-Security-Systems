import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import sounddevice as sd
import wave
from flask import Flask, render_template, request, jsonify
from sklearn.preprocessing import LabelEncoder
from playsound import playsound
from twilio.rest import Client

# Initialize Flask app
app = Flask(__name__)

# Load YAMNet model from TensorFlow Hub
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

# Load trained model
model = tf.keras.models.load_model('trained_model.h5')

# Load label encoder classes
label_classes = np.load('label_classes.npy', allow_pickle=True)

# Twilio Configuration
TWILIO_ACCOUNT_SID = 'ACf986c62ffae4902bbc5771b2c6906a9c'
TWILIO_AUTH_TOKEN = '4c91e709828579a556cc2c73d810193e'
TWILIO_PHONE_NUMBER = '+16814122086'
RECIPIENT_PHONE_NUMBER = '+918660486877'

# Function to record audio and save as a .wav file
def record_audio(filename, duration=5, samplerate=16000):
    print("Recording audio...")
    audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()
    with wave.open(filename, 'wb') as wavfile:
        wavfile.setnchannels(1)  # Mono channel
        wavfile.setsampwidth(2)  # 16-bit depth
        wavfile.setframerate(samplerate)
        wavfile.writeframes((audio_data * 32767).astype(np.int16))  # Convert to 16-bit PCM
    print(f"Recording saved to {filename}")

# Function to extract YAMNet embeddings
def extract_yamnet_embeddings(audio_file, embedding_size=1024):
    waveform, sr = librosa.load(audio_file, sr=16000)
    _, embeddings, _ = yamnet_model(waveform)
    flattened_embedding = tf.reshape(embeddings, [-1]).numpy()
    embedding_length = flattened_embedding.shape[0]
    if embedding_length < embedding_size:
        padding = np.zeros((embedding_size - embedding_length,))
        padded_embedding = np.concatenate([flattened_embedding, padding], axis=0)
    else:
        padded_embedding = flattened_embedding[:embedding_size]
    return padded_embedding.reshape(1, -1)

# Function to send SMS via Twilio
def send_sms_notification():
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    message = client.messages.create(
        body="Unsafe sound detected! Please check your system immediately.",
        from_=TWILIO_PHONE_NUMBER,
        to=RECIPIENT_PHONE_NUMBER
    )
    print(f"SMS sent: SID {message.sid}")

# Function to make a call via Twilio
def make_call():
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    call = client.calls.create(
        twiml='<Response><Say>Alert! Unsafe sound detected at your location. Please respond immediately.</Say></Response>',
        from_=TWILIO_PHONE_NUMBER,
        to=RECIPIENT_PHONE_NUMBER
    )
    print(f"Call initiated: SID {call.sid}")

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/record', methods=['POST'])
def record():
    # Save recorded audio in the static folder
    audio_file = os.path.join('static', 'recorded_audio.wav')
    record_audio(audio_file, duration=5)
    return jsonify({'message': 'Audio recorded successfully', 'audio_file': audio_file})

@app.route('/predict', methods=['POST'])
def predict():
    # Load the recorded audio file and predict the label
    audio_file = os.path.join('static', 'recorded_audio.wav')
    if not os.path.exists(audio_file):
        return jsonify({'error': 'No audio file found. Please record first.'})
    
    # Extract embeddings and predict label
    embeddings = extract_yamnet_embeddings(audio_file)
    prediction = model.predict(embeddings)
    predicted_label = label_classes[np.argmax(prediction)]

    # If the label is "unsafe", trigger alerts
    if predicted_label == "unsafe":
        send_sms_notification()
        make_call()

    return jsonify({'predicted_label': predicted_label})

if __name__ == '__main__':
    app.run(debug=True)
