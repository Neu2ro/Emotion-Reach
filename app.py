from flask import Flask, render_template, request
import matplotlib.pyplot as plt
from tempfile import NamedTemporaryFile
import librosa
import numpy as np
import pydub
import pickle

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

clf = pickle.load(open('model.pkl','rb'))
scaler = pickle.load(open('scaler.pkl','rb'))
app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predictOutput():
    audfile = request.files['audfile']
    # aud_path = "./audio/"+audfile.mp3
    # audfile.save(aud_path)
    a=()
    a=predict(audfile)
    emo=a[0]
    prob=[a[1],a[2],a[3],a[4]]
    return render_template('index.html', prediction=emo, prob=prob)

def predict(audfile):
    with NamedTemporaryFile(delete=False) as temp:
        audfile.save(temp.name)
        y, sr = librosa.load(temp.name, sr=None)
        features = extract_feature(temp.name, True, True, True)
    X = scaler.transform([features])
    emotion_labels = ['angry 😡', 'sad 😔', 'neutral 🙂', 'happy 😊']
    proba = clf.predict_proba(X)[0]
    probability = list(proba)
    emotion_idx = np.argmax(proba)
    predicted_emotion = emotion_labels[emotion_idx]

    # Image(filename="/Users/ritikahotwani/Downloads/MLProject/angimage.png")

  
    prob_angry = round(probability[0] * 100, 4)
    prob_sad = round(probability[1] * 100, 4)
    prob_neutral = round(probability[2] * 100, 4)
    prob_happy = round(probability[3] * 100, 4)
    return (predicted_emotion, prob_angry, prob_sad, prob_neutral, prob_happy)


def extract_feature(file_name, mfcc, chroma, mel):
    X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    if chroma:
        stft=np.abs(librosa.stft(X))
    result=np.array([])
    if mfcc:
        mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=50).T, axis=0)
        result=np.hstack((result, mfccs))
    if chroma:
        chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        result=np.hstack((result, chroma))
    if mel:
        mel=np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T,axis=0)
        result=np.hstack((result, mel))
    return result


if __name__ == '__main__':
    app.run(port=3000, debug=True)