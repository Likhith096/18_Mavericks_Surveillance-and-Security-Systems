# import os
# import random
# import glob
# import numpy as np
# import tensorflow as tf
# import tensorflow_hub as hub
# import librosa
# import sounddevice as sd
# import wave
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from tensorflow.keras.models import Sequential, Model
# from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
# from tensorflow.keras.callbacks import EarlyStopping


# seed_value = 42
# np.random.seed(seed_value)
# random.seed(seed_value)
# tf.random.set_seed(seed_value)

# # Load YAMNet model from TensorFlow Hub
# yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

# # Path to the dataset directory
# dataset_dir = 'audioset'

# # Function to get all audio file paths and labels
# def get_audio_files_and_labels(dataset_dir):
#     audio_files = []
#     labels = []
#     for label_dir in os.listdir(dataset_dir):
#         label_path = os.path.join(dataset_dir, label_dir)
#         if os.path.isdir(label_path):
#             wav_files = glob.glob(os.path.join(label_path, '*.wav'))
#             for wav_file in wav_files:
#                 audio_files.append(wav_file)
#                 labels.append(label_dir)
#     return audio_files, labels

# # Get all audio files and labels
# audio_files, labels = get_audio_files_and_labels(dataset_dir)

# # Function to extract YAMNet embeddings and ensure consistent size
# def extract_yamnet_embeddings(audio_files_batch, embedding_size=1024):
#     embeddings_batch = []
#     for file in audio_files_batch:
#         # Load the audio file using librosa
#         waveform, sr = librosa.load(file, sr=16000)  # Load at 16 kHz (model's expected sample rate)
        
#         # Apply pitch-shifting (data augmentation) correctly
#         waveform = librosa.effects.pitch_shift(waveform, sr=sr, n_steps=2)  # Corrected this line

#         # Ensure the waveform is 1-dimensional
#         waveform = waveform.reshape(-1)
        
#         # Make predictions using YAMNet model
#         scores, embeddings, spectrogram = yamnet_model(waveform)

#         # Flatten embeddings to a single vector
#         flattened_embedding = tf.reshape(embeddings, [-1]) 
#         flattened_embedding = flattened_embedding.numpy()

#         # Pad or truncate to the fixed embedding size
#         embedding_length = flattened_embedding.shape[0]
#         if embedding_length < embedding_size:
#             padding = np.zeros((embedding_size - embedding_length,))
#             padded_embedding = np.concatenate([flattened_embedding, padding], axis=0)
#         else:
#             padded_embedding = flattened_embedding[:embedding_size]
        
#         embeddings_batch.append(padded_embedding)
    
#     return np.array(embeddings_batch)

# # Batch generator for extracting embeddings
# def batch_generator(audio_files, batch_size=32):
#     for i in range(0, len(audio_files), batch_size):
#         yield audio_files[i:i+batch_size]

# # Process embeddings in batches
# embeddings = []
# for batch in batch_generator(audio_files, batch_size=32):
#     embeddings_batch = extract_yamnet_embeddings(batch)
#     embeddings.append(embeddings_batch)

# # Flatten the list of embeddings
# embeddings = np.concatenate(embeddings, axis=0)

# # Encode labels - Fit the encoder on the entire set of labels
# label_encoder = LabelEncoder()
# encoded_labels = label_encoder.fit_transform(labels)

# # Split data for training and testing
# X_train, X_test, y_train, y_test = train_test_split(embeddings, encoded_labels, test_size=0.2, random_state=42)

# # Corrected Model Architecture
# model_input = tf.keras.Input(shape=(embeddings.shape[1],)) # Input layer
# x = Dense(512, activation='relu')(model_input)
# x = Dropout(0.5)(x)

# # Reshaping the input for Conv1D
# x = tf.keras.layers.Reshape((x.shape[1], 1))(x)  # Reshaping to (batch_size, sequence_length, 1)

# # Now you can apply Conv1D layer
# x = Conv1D(64, 3, activation='relu')(x)
# x = MaxPooling1D(2)(x)

# x = Dense(128, activation='relu')(x)
# x = Dropout(0.5)(x)
# x = Flatten()(x)  # Flattening after convolutional layers

# # Output layer
# output = Dense(len(np.unique(y_train)), activation='softmax')(x)

# # Build and compile model
# model = Model(inputs=model_input, outputs=output)
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# # Set up early stopping
# early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# # Train the model
# model.fit(X_train, y_train, epochs=30, batch_size=10, validation_data=(X_test, y_test), callbacks=[early_stopping])

# # Evaluate the model
# loss, accuracy = model.evaluate(X_test, y_test)
# print(f"Test Accuracy: {accuracy:.2f}")


# # Function to record audio and save as a .wav file
# def record_audio(filename, duration=5, samplerate=16000):
#     print(f"Recording for {duration} seconds...")
#     audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
#     sd.wait()  # Wait for recording to finish
#     wavfile = wave.open(filename, 'wb')
#     wavfile.setnchannels(1)  # Mono channel
#     wavfile.setsampwidth(2)  # 16-bit depth
#     wavfile.setframerate(samplerate)
#     wavfile.writeframes((audio_data * 32767).astype(np.int16))  # Convert to 16-bit PCM
#     wavfile.close()
#     print(f"Recording saved to {filename}")

# # Function to make a prediction from an audio file
# def predict_audio(filename):
#     # Extract YAMNet embeddings for the recorded audio
#     waveform, sr = librosa.load(filename, sr=16000)
#     embeddings = extract_yamnet_embeddings([filename])
    
#     # Make prediction
#     prediction = model.predict(embeddings)
#     predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]
#     print(f"Predicted Label for {filename}: {predicted_label}")

# # Record and predict in real-time
# if __name__ == "__main__":
#     while True:
#         # Record audio for 5 seconds
#         record_audio('recorded_audio.wav', duration=5)
        
#         # Predict if the recorded audio is safe or unsafe
#         predict_audio('recorded_audio.wav')
        
#         break



import os
import random
import glob
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import sounddevice as sd
import wave
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Reshape
from tensorflow.keras.callbacks import EarlyStopping

# Set random seed
seed_value = 42
np.random.seed(seed_value)
random.seed(seed_value)
tf.random.set_seed(seed_value)

# Load YAMNet model from TensorFlow Hub
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

# Path to the dataset directory
dataset_dir = 'audioset'

# Data augmentation: Adding noise and pitch-shifting
def augment_audio(waveform, sr):
    noise_factor = 0.005  # Low noise level
    waveform += noise_factor * np.random.normal(size=waveform.shape)  # Add noise
    waveform = librosa.effects.pitch_shift(waveform, sr=sr, n_steps=random.choice([-2, 2]))  # Random pitch shift
    return waveform

# Function to get all audio files and labels
def get_audio_files_and_labels(dataset_dir):
    audio_files = []
    labels = []
    for label_dir in os.listdir(dataset_dir):
        label_path = os.path.join(dataset_dir, label_dir)
        if os.path.isdir(label_path):
            wav_files = glob.glob(os.path.join(label_path, '*.wav'))
            for wav_file in wav_files:
                audio_files.append(wav_file)
                labels.append(label_dir)
    return audio_files, labels

# Get all audio files and labels
audio_files, labels = get_audio_files_and_labels(dataset_dir)

# Function to extract YAMNet embeddings and ensure consistent size
def extract_yamnet_embeddings(audio_files_batch, embedding_size=1024):
    embeddings_batch = []
    for file in audio_files_batch:
        waveform, sr = librosa.load(file, sr=16000)
        waveform = augment_audio(waveform, sr)
        waveform = waveform.reshape(-1)
        
        # Get embeddings
        _, embeddings, _ = yamnet_model(waveform)
        flattened_embedding = tf.reshape(embeddings, [-1]).numpy()
        padded_embedding = np.pad(flattened_embedding, (0, max(0, embedding_size - flattened_embedding.shape[0])))
        embeddings_batch.append(padded_embedding[:embedding_size])
    return np.array(embeddings_batch)

# Process embeddings in batches
embeddings = []
for batch in range(0, len(audio_files), 32):
    embeddings_batch = extract_yamnet_embeddings(audio_files[batch:batch+32])
    embeddings.append(embeddings_batch)
embeddings = np.concatenate(embeddings, axis=0)

# Normalize embeddings
scaler = StandardScaler()
embeddings = scaler.fit_transform(embeddings)

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(embeddings, encoded_labels, test_size=0.2, random_state=42)

# Model architecture
model_input = tf.keras.Input(shape=(embeddings.shape[1],))
x = Dense(256, activation='relu')(model_input)
x = Dropout(0.4)(x)
x = Reshape((x.shape[1], 1))(x)
x = Conv1D(64, 3, activation='relu')(x)
x = MaxPooling1D(2)(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(len(np.unique(y_train)), activation='softmax')(x)

# Compile model
model = Model(inputs=model_input, outputs=output)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train model
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")

# Function to record audio and save as a .wav file
def record_audio(filename, duration=5, samplerate=16000):
    print(f"Recording for {duration} seconds...")
    audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()  # Wait for recording to finish
    wavfile = wave.open(filename, 'wb')
    wavfile.setnchannels(1)  # Mono channel
    wavfile.setsampwidth(2)  # 16-bit depth
    wavfile.setframerate(samplerate)
    wavfile.writeframes((audio_data * 32767).astype(np.int16))  # Convert to 16-bit PCM
    wavfile.close()
    print(f"Recording saved to {filename}")

# Function to make a prediction from an audio file
def predict_audio(filename):
    waveform, sr = librosa.load(filename, sr=16000)
    embeddings = extract_yamnet_embeddings([filename])
    
    prediction = model.predict(embeddings)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]
    print(f"Predicted Label for {filename}: {predicted_label}")
    

# Record and predict in real-time
if __name__ == "__main__":
    while True:
        record_audio('recorded_audio.wav', duration=5)
        predict_audio('recorded_audio.wav')
        break
