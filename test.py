import os
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import numpy as np

# Load the YAMNet model from TensorFlow Hub
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

# Load class labels (YAMNet's categories)
class_map_path = yamnet_model.class_map_path().numpy().decode('utf-8')
class_labels = [line.strip() for line in open(class_map_path).readlines()]

# Unsafe categories and their threat levels
unsafe_classes = {
    "Explosion": "high",
    "Gunshot, gunfire": "high",
    "Screaming": "high",
    "Crying, sobbing": "medium",
    "Glass breaking": "high",
    "Emergency vehicle (siren)": "medium",
    "Alarm": "medium"
}

# Function to load and preprocess audio file
def load_audio(file_path):
    # Load audio with librosa (16kHz is the expected sample rate for YAMNet)
    waveform, sample_rate = librosa.load(file_path, sr=16000)
    return waveform

# Function to classify audio
def classify_sound(file_path):
    waveform = load_audio(file_path)
    # Run YAMNet prediction
    scores, embeddings, spectrogram = yamnet_model(waveform)
    
    # Get the class with the highest average score
    prediction = scores.numpy().mean(axis=0)
    top_class_index = np.argmax(prediction)
    top_class = class_labels[top_class_index]
    confidence = prediction[top_class_index]
    
    # Check if the sound is "unsafe" or "safe"
    if top_class in unsafe_classes:
        threat_level = unsafe_classes[top_class]
        safety_status = "unsafe"
    else:
        threat_level = "low"
        safety_status = "safe"
    
    # Output results
    return {
        "file": file_path,
        "detected_sound": top_class,
        "confidence": confidence,
        "safety_status": safety_status,
        "threat_level": threat_level
    }

# Function to process a dataset folder
def process_dataset(folder_path):
    results = []
    # Iterate through all .wav files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.wav'):  # Only process .wav files
            file_path = os.path.join(folder_path, file_name)
            result = classify_sound(file_path)
            results.append(result)
            # Print the result for each file
            print(f"File: {result['file']}")
            print(f"  Detected Sound: {result['detected_sound']} (Confidence: {result['confidence']:.2f})")
            print(f"  Safety Status: {result['safety_status']}")
            print(f"  Threat Level: {result['threat_level']}")
            print("-" * 50)
    return results

# Example usage
dataset_folder = 'audioset/test'  # Replace with the path to your folder containing .wav files
results = process_dataset(dataset_folder)
