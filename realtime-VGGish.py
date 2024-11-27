# import os
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# import glob
# import numpy as np
# import tensorflow as tf
# tf.get_logger().setLevel('ERROR')
# import tensorflow_hub as hub
# import librosa
# import sounddevice as sd
# import soundfile as sf  # Added to save the recorded file
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization

# # Load VGGish model from TensorFlow Hub
# vggish_model = hub.load('https://tfhub.dev/google/vggish/1')

# # Path to the dataset directory
# dataset_dir = 'audioset'  # Replace with your dataset path

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

# # Function to extract VGGish embeddings for a batch of audio files
# def extract_embeddings_batch(file_paths):
#     embeddings_batch = []
#     for file_path in file_paths:
#         waveform, sr = librosa.load(file_path, sr=16000)
#         waveform = waveform[:sr * 10]
#         waveform = np.expand_dims(waveform, axis=0)
#         waveform_tensor = tf.squeeze(tf.convert_to_tensor(waveform, dtype=tf.float32))
#         embeddings = vggish_model(waveform_tensor)
#         embeddings_batch.append(embeddings.numpy().mean(axis=0))
#     return np.array(embeddings_batch)

# # Batch generator for extracting embeddings
# def batch_generator(audio_files, batch_size=32):
#     for i in range(0, len(audio_files), batch_size):
#         yield audio_files[i:i+batch_size]

# # Process embeddings in batches
# embeddings = []
# for batch in batch_generator(audio_files, batch_size=32):
#     embeddings_batch = extract_embeddings_batch(batch)
#     embeddings.append(embeddings_batch)

# # Flatten the list of embeddings
# embeddings = np.concatenate(embeddings, axis=0)

# # Encode labels
# label_encoder = LabelEncoder()
# encoded_labels = label_encoder.fit_transform(labels)

# # Split data for training and testing
# X_train, X_test, y_train, y_test = train_test_split(embeddings, encoded_labels, test_size=0.2, random_state=42)

# # Build CNN model for audio classification
# model = Sequential([
#     Conv1D(64, 3, activation='relu', input_shape=(128, 1)),
#     MaxPooling1D(2),
#     Conv1D(128, 3, activation='relu'),
#     MaxPooling1D(2),
#     Dropout(0.5),
#     Flatten(),
#     Dense(128, activation='relu'),
#     BatchNormalization(),
#     Dense(len(np.unique(y_train)), activation='softmax')
# ])

# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.summary()

# # Reshape data to fit the CNN input
# X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
# X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# # Train the model
# model.fit(X_train_cnn, y_train, epochs=20, batch_size=10, validation_split=0.1)

# # Evaluate the model
# loss, accuracy = model.evaluate(X_test_cnn, y_test)
# print(f"Test Accuracy: {accuracy:.2f}")

# # Prediction on a new audio file
# def predict_audio(file_path):
#     embedding = extract_embeddings_batch([file_path]).reshape(1, -1, 1)  # Reshape to (1, features, 1)
#     prediction = model.predict(embedding)
#     predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
#     print("Prediction:")
#     print(f"Detected Sound: {predicted_label[0]}")

# # Function to record sound from microphone, save the file, and make a prediction
# def record_and_predict(duration=5, samplerate=16000, file_name="recorded_audio.wav"):
#     print(f"Recording for {duration} seconds...")
    
#     # Record audio from the microphone
#     recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
#     sd.wait()  # Wait until recording is finished
    
#     # Save the recorded audio to a file
#     sf.write(file_name, recording, samplerate)
#     print(f"Audio saved as {file_name}")
    
#     # Predict using the saved audio file
#     predict_audio(file_name)

# # Example usage for real-time prediction
# record_and_predict(duration=5)  # Record for 5 seconds


import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow_hub as hub
import librosa
import sounddevice as sd
import soundfile as sf  # Added to save the recorded file
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

# Load VGGish model from TensorFlow Hub
vggish_model = hub.load('https://tfhub.dev/google/vggish/1')

# Path to the dataset directory
dataset_dir = 'audioset'  # Replace with your dataset path

# Function to get all audio file paths and labels
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

# Function to extract VGGish embeddings for a batch of audio files
def extract_embeddings_batch(file_paths):
    embeddings_batch = []
    for file_path in file_paths:
        waveform, sr = librosa.load(file_path, sr=16000)
        waveform = waveform[:sr * 10]  # Ensure it's a fixed-length audio (10 seconds)
        waveform = np.expand_dims(waveform, axis=0)
        waveform_tensor = tf.squeeze(tf.convert_to_tensor(waveform, dtype=tf.float32))
        embeddings = vggish_model(waveform_tensor)
        embeddings_batch.append(embeddings.numpy().mean(axis=0))
    return np.array(embeddings_batch)

# Batch generator for extracting embeddings
def batch_generator(audio_files, batch_size=32):
    for i in range(0, len(audio_files), batch_size):
        yield audio_files[i:i+batch_size]

# Process embeddings in batches
embeddings = []
for batch in batch_generator(audio_files, batch_size=32):
    embeddings_batch = extract_embeddings_batch(batch)
    embeddings.append(embeddings_batch)

# Flatten the list of embeddings
embeddings = np.concatenate(embeddings, axis=0)

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(embeddings, encoded_labels, test_size=0.2, random_state=42)

# Build CNN model for audio classification with more layers
model = Sequential([
    Conv1D(64, 5, activation='relu', input_shape=(128, 1)),
    MaxPooling1D(2),
    BatchNormalization(),
    Conv1D(128, 5, activation='relu'),
    MaxPooling1D(2),
    Dropout(0.5),
    BatchNormalization(),
    Conv1D(256, 3, activation='relu'),
    MaxPooling1D(2),
    Dropout(0.5),
    Flatten(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(len(np.unique(y_train)), activation='softmax')
])

# Compile the model with Adam optimizer and learning rate decay
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Reshape data to fit the CNN input
X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Set up early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# Train the model with increased epochs and callbacks
model.fit(X_train_cnn, y_train, epochs=50, batch_size=10, validation_split=0.1, callbacks=[early_stopping, reduce_lr])

# Evaluate the model
loss, accuracy = model.evaluate(X_test_cnn, y_test)
print(f"Test Accuracy: {accuracy:.2f}")

# Prediction on a new audio file
def predict_audio(file_path):
    embedding = extract_embeddings_batch([file_path]).reshape(1, -1, 1)  # Reshape to (1, features, 1)
    prediction = model.predict(embedding)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
    print("Prediction:")
    print(f"Detected Sound: {predicted_label[0]}")

# Function to record sound from microphone, save the file, and make a prediction
def record_and_predict(duration=5, samplerate=16000, file_name="recorded_audio.wav"):
    print(f"Recording for {duration} seconds...")
    
    # Record audio from the microphone
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished
    
    # Save the recorded audio to a file
    sf.write(file_name, recording, samplerate)
    print(f"Audio saved as {file_name}")
    
    # Predict using the saved audio file
    predict_audio(file_name)

# Example usage for real-time prediction
record_and_predict(duration=5)  # Record for 5 seconds


# Prediction on test data and printing file names along with predicted and true labels


for i in range(len(X_test)):
    # Get the true label for this test sample
    true_label = label_encoder.inverse_transform([y_test[i]])[0]

    # Reshape the data point to match the input shape of the model
    embedding = X_test[i].reshape(1, -1, 1)
    
    # Make prediction
    prediction = model.predict(embedding)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]

    # Get the filename corresponding to the test instance
    test_file = audio_files[i]  # Directly access the file from audio_files

    # Print the results
    print(f"File: {test_file}")
    print(f"True Label: {true_label}")
    print(f"Predicted Label: {predicted_label}")
    print("-" * 40)