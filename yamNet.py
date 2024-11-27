import tensorflow as tf
import tensorflow_hub as hub
import librosa
import numpy as np
import os

# Load the YAMNet model from TensorFlow Hub
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

# Load class labels (YAMNet's categories)
class_map_path = yamnet_model.class_map_path().numpy().decode('utf-8')
class_labels = [line.strip() for line in open(class_map_path).readlines()]

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
    
    # Get the class with highest score
    prediction = scores.numpy().mean(axis=0)
    top_class_index = np.argmax(prediction)
    
    # Print the predicted class, file name, and its score
    print(f"File: {os.path.basename(file_path)}")
    print(f"Detected Sound: {class_labels[top_class_index]} (Confidence: {prediction[top_class_index]:.2f})\n")

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
