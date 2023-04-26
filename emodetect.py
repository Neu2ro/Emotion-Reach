import resampy
import librosa
import soundfile
import os
import glob
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.utils.multiclass import unique_labels
import pandas as pd

def extract_feature(file_name, mfcc, chroma, mel):
    X, sample_rate = librosa.load(os.path.join(file_name), res_type='kaiser_fast')
    if chroma:
        stft=np.abs(librosa.stft(X))
    result=np.array([])
    if mfcc:
        mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=20).T, axis=0)
        result=np.hstack((result, mfccs))
    if chroma:
        chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        result=np.hstack((result, chroma))
    if mel:
        mel=np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T,axis=0)
        result=np.hstack((result, mel))
    return result

# Emotions in the RAVDESS & TESS dataset
emotions={
  '01':'angry',
  '02':'sad',
  '03':'neutral',
  '04':'happy',
}
observed_emotions=['angry','sad','neutral','happy']

def load_data(test_size=0.2):
    x,y=[],[]
    for file in glob.glob('/content/drive/MyDrive/Dataset2/Actor_*/*.mp3'):
        file_name = os.path.basename(file)
        emotion = emotions[file_name.split("-")[1]]
        if emotion not in observed_emotions:
            continue
        feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
    return train_test_split(np.array(x), y, test_size = test_size, train_size= 0.5)
    #  random_state = 9

# Split the dataset
import time
x_train, x_test, y_train, y_test = load_data(test_size = 0.25)

#Get the shape of the training and testing datasets
print((x_train.shape[0], x_test.shape[0]))

# Get the number of features extracted
print(f'\n\nFeatures extracted: {x_train.shape[1]}')

# Initialize the Multi Layer Perceptron Classifier
model=MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)

# Train the model
model.fit(x_train, y_train)
joblib.dump(model, 'mlp_classifier.joblib')

# Create an instance of StandardScaler and fit it to the training data
scaler = StandardScaler()
scaler.fit(x_train)
# Save the StandardScaler object to a file using joblib.dump
joblib.dump(scaler, 'scaler.joblib')

# Predict for the test set
y_pred = model.predict(x_test)

# Calculate the accuracy of our model
accuracy = accuracy_score(y_true = y_test, y_pred = y_pred)
# Print the accuracy
print("\n\nAccuracy: {:.2f}%\n\n".format(accuracy*100))

print(classification_report(y_test, y_pred))

# matrix = confusion_matrix(y_test,y_pred)
# print(matrix)

def plot(y_true, y_pred):
  labels = unique_labels(y_test)
  column = [f'Predicted({label})' for label in labels]
  indices = [f'Actual({label})' for label in labels]
  table = pd.DataFrame(confusion_matrix(y_true, y_pred),
                       columns = column, index = indices)
  return table

# Load the pre-trained MLP classifier and StandardScaler object
clf = joblib.load('mlp_classifier.joblib')
scaler = joblib.load('scaler.joblib')

# Load the single audio file and extract features
audio_file = '/content/drive/MyDrive/Dataset2/Actor_ang/angry-01-12.mp3'
y, sr = librosa.load(audio_file, sr=None)
features = extract_feature(audio_file, mfcc=True, chroma=True, mel=True)

# Normalize the extracted features
X = scaler.transform([features])

# Make a prediction on the normalized features
emotion_labels = ['angry','sad','neutral','happy']
proba = clf.predict_proba(X)[0]
emotion_idx = np.argmax(proba)
predicted_emotion = emotion_labels[emotion_idx]

print('\n\nPredicted emotion:', predicted_emotion)
print('Probability distribution:', proba)
