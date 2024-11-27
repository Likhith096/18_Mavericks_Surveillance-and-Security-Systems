import tensorflow as tf
import tensorflow_hub as hub
import librosa
import numpy as np
import os
import re

# Load the YAMNet model from TensorFlow Hub
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

# Load class labels (YAMNet's categories)
class_map_path = yamnet_model.class_map_path().numpy().decode('utf-8')

# Extract only the display_name (second item in each line, assuming CSV-like format)
class_labels = [line.strip().split(",")[2].strip('"') for line in open(class_map_path).readlines()]

# Define distress or unsafe classes (these are examples, you can customize this list based on your needs)
unsafe_classes = [
    "Air horn", "cry", "Explosion", "chuckle", "smashing", "Snort", 
    "shout", "scream", "Gunshot", "drilling", "Glass break", "crack"
    # Add other classes that represent distress or danger
]

# Function to preprocess class label (ignore suffixes like 'ing' or 'n')
def preprocess_label(label):
    # Remove suffixes like 'ing', 'n', etc.
    label = label.lower()
    label = re.sub(r'(ing|n)$', '', label)  # Remove 'ing' or 'n' from the end of the word
    return label

# Function to load and preprocess audio file
def load_audio(file_path):
    # Load audio with librosa (16kHz is the expected sample rate for YAMNet)
    waveform, sample_rate = librosa.load(file_path, sr=16000)
    return waveform

# Function to classify threat level based on confidence score
def classify_threat_level(confidence):
    if confidence >= 0.7:
        return "High"
    elif confidence >= 0.4:
        return "Medium"
    else:
        return "Low"

# Function to classify audio
def classify_sound(file_path):
    waveform = load_audio(file_path)
    # Run YAMNet prediction
    scores, embeddings, spectrogram = yamnet_model(waveform)
    
    # Get the class with the highest score
    prediction = scores.numpy().mean(axis=0)
    top_class_index = np.argmax(prediction)
    top_class_label = class_labels[top_class_index]
    confidence = prediction[top_class_index]
    
    # Preprocess the class label and match with unsafe classes
    processed_label = preprocess_label(top_class_label)
    status = "Safe"
    threat_level = None
    
    for unsafe_class in unsafe_classes:
        if preprocess_label(unsafe_class) in processed_label:
            status = "Unsafe"
            threat_level = classify_threat_level(confidence)  # Assign threat level if unsafe sound detected
            break
    
    # Print the predicted class (display_name), file name, and its score
    print(f"File: {os.path.basename(file_path)}")
    print(f"Detected Sound: {top_class_label} (Confidence: {confidence:.2f})")
    print(f"Status: {status}")
    
    if status == "Unsafe":
        print(f"Threat Level: {threat_level}")
    
    print("\n")

# Function to process all audio files in a folder
def process_folder(folder_path):
    # Get all audio files in the folder (e.g., .wav, .mp3 files)
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if file_name.endswith('.wav'):  # Adjust this for your audio format (e.g., .mp3)
            classify_sound(file_path)

# Example usage
audio_folder = 'audioset/danger'  # Path to your folder containing audio files
process_folder(audio_folder)
