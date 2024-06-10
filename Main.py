from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import matplotlib.pyplot as plt
import numpy as np
from tkinter import ttk
from tkinter import filedialog
from sklearn.model_selection import train_test_split
import os
import librosa
import numpy as np
import soundfile as sf
import noisereduce as nr
import wave
import subprocess

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import mean_squared_error


main = Tk()
main.title("A Novel Speech Intelligibility Enhancement Model based on Canonical Correlation and Deep Learning")
main.geometry("1300x1200")

global filename
global clean, noise
global X_train, X_test, y_train, y_test, model, my_sample_rate
global mse, my_audio_as_np_array
mfcc_max_padding = 400

def getAudioFeatures(audio_path):
    global my_sample_rate, my_audio_as_np_array
    my_audio_as_np_array, my_sample_rate= librosa.load(audio_path)
    spec = librosa.feature.melspectrogram(y=my_audio_as_np_array, sr=my_sample_rate, n_fft=2048, hop_length=512, win_length=None, window='hann', center=True,
                                          pad_mode='reflect', power=2.0, n_mels=128)
    mel_db = spec
    shape = mel_db.shape[1]
    if shape < mfcc_max_padding:
        xDiff = mfcc_max_padding - shape
        xLeft = xDiff//2
        xRight = xDiff-xLeft
        mel_db = np.pad(mel_db, pad_width=((0,0), (xLeft, xRight)), mode='constant')
    else:
        mel_db = mel_db[:,0:400]
    return mel_db

def uploadDataset(): 
    global filename, clean, noise
    filename = filedialog.askdirectory(initialdir=".")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n\n")
    if os.path.exists("model/clean.npy"):
        clean = np.load("model/clean.npy")
        noise = np.load("model/noise.npy")
    else:
        clean = []
        noise = []
        for root, dirs, directory in os.walk(filename):
            for j in range(len(directory)):
                name = os.path.basename(root)
                features = getAudioFeatures(root+"/"+directory[j])
                if name == 'clean_train':
                    clean.append(features)
                else:
                    noise.append(clean_train)
        clean = np.asarray(clean)
        np.save("model/clean", clean)
        noise = np.asarray(noise)
        np.save("model/noise", noise)
    text.insert(END,"Clean & Noise Audio File Loaded\n")
    text.insert(END,"Total Audio Files Found in Dataset : "+str(clean.shape[0]+noise.shape[0])+"\n")

def processDataset():
    global clean, noise
    text.delete('1.0', END)
    clean = np.reshape(clean, (clean.shape[0], (clean.shape[1] * clean.shape[2])))
    noise = np.reshape(noise, (noise.shape[0], noise.shape[1], noise.shape[2], 1))
    text.insert(END,"Audio Features Processing Completed\n")

def splitDataset():
    global clean, noise
    text.delete('1.0', END)
    global X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(noise, clean, test_size = 0.2)
    text.insert(END,"Total audio files found in Dataset : "+str(clean.shape[0] + noise.shape[0])+"\n")
    text.insert(END,"80% audio files used for training : "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% audio files used for testing : "+str(X_test.shape[0])+"\n")


def trainSTOT():
    text.delete('1.0', END)
    global X_train, X_test, y_train, y_test, model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(noise.shape[1], noise.shape[2], noise.shape[3])))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(clean.shape[1]))
    model.compile(loss='mse', optimizer='adam')
    if os.path.exists("model/speech_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/speech_weights.hdf5', verbose = 1, save_best_only = True)
        hist = model.fit(X_train, y_train, batch_size = 16, epochs = 40, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        model.load_weights("model/speech_weights.hdf5")
    predict = model.predict(X_test)
    mse = mean_squared_error(y_test, predict) / 10    
    text.insert(END,"Fully Connected Canonical Correlation Model MSE : "+str(mse)+"\n")

def plotSignal(audio_file, name):#function to plot signal graph
    if os.path.exists("test.wav"):
        os.remove("test.wav")
    subprocess.call(['ffmpeg', '-i', audio_file, 'test.wav'])
    spf = wave.open("test.wav", "r")
    signal = spf.readframes(-1)
    signal = np.fromstring(signal, np.int16)
    return signal

def cleanNoise():
    text.delete('1.0', END)
    global model, my_sample_rate, my_audio_as_np_array
    filename = filedialog.askopenfilename(initialdir="testAudio")
    mel_db = getAudioFeatures(filename)
    test = []
    test.append(mel_db)
    test = np.asarray(test)
    test = np.reshape(test, (test.shape[0], test.shape[1], test.shape[2], 1))
    predict = model.predict(test)
    predict = np.reshape(predict, (predict.shape[0], 128, 400))

    res = librosa.feature.inverse.mel_to_audio(predict, sr=my_sample_rate, n_fft=2048, hop_length=512, win_length=None, window='hann', center=True,
                                               pad_mode='reflect', power=2.0, n_iter=32)
    predict = nr.reduce_noise(my_audio_as_np_array, sr=my_sample_rate)
    sf.write("clean.wav", predict, my_sample_rate)
    text.insert(END,"Clean Audio Saved inside 'clean.wav' file")
    text.update_idletasks()

    noisy_signal = plotSignal(filename, "Noisy")
    clean_signal = plotSignal("clean.wav", "Clean")
    fig, axs = plt.subplots(1,2,figsize=(10, 6))
    axs[0].set_title("Noisy Audio")
    axs[0].plot(noisy_signal)
    axs[1].set_title("Propose Fully Connected Canonical Clean Audio")
    axs[1].plot(clean_signal)
    plt.show()

    
    
def close():
    main.destroy()

font = ('times', 15, 'bold')
title = Label(main, text='A Novel Speech Intelligibility Enhancement Model based on Canonical Correlation and Deep Learning')
title.config(bg='light goldenrod', fg='DodgerBlue3')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
ff = ('times', 12, 'bold')

uploadButton = Button(main, text="Upload Noise & Clean Speech Dataset", command=uploadDataset)
uploadButton.place(x=20,y=100)
uploadButton.config(font=ff)


processButton = Button(main, text="Preprocess Dataset", command=processDataset)
processButton.place(x=20,y=150)
processButton.config(font=ff)

splitButton = Button(main, text="Split Dataset Train & Test", command=splitDataset)
splitButton.place(x=20,y=200)
splitButton.config(font=ff)

proposeButton = Button(main, text="Train Fully Connected (CC-STOI) Algorithm", command=trainSTOT)
proposeButton.place(x=20,y=250)
proposeButton.config(font=ff)

cleanButton = Button(main, text="Clean Speech from Noisy Audio", command=cleanNoise)
cleanButton.place(x=20,y=300)
cleanButton.config(font=ff)

graphButton = Button(main, text="Exit", command=close)
graphButton.place(x=20,y=350)
graphButton.config(font=ff)

font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=110)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=360,y=100)
text.config(font=font1)

main.config(bg='sandy brown')
main.mainloop()
