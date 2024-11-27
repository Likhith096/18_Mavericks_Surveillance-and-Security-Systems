import tensorflow as tf
import tensorflow_hub as hub
import librosa
import numpy as np

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
    
    # Print the predicted class and its score
    print(f"Detected Sound: {class_labels[top_class_index]} (Confidence: {prediction[top_class_index]:.2f})")

# Example usage
audio_file = 'audioset/normal/tvsound.wav'
classify_sound(audio_file)
