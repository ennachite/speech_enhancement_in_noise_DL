import glob
import librosa
import soundfile as sf
import random
import os
from sklearn import preprocessing

def alpha(RSB):
    return 10 ** (-RSB / 20)

if __name__ == '__main__':
    # Get the list of all the audio files
    audioFiles = glob.glob("data/LibriSpeech/dev-clean/**/**/*.flac", recursive=False)
    # Get the list of all the noise files
    noiseFile = glob.glob("data/noise/babble.wav")
    noise, sr = librosa.load(noiseFile[0])
    i = 0
    x = len(audioFiles)
    if not os.path.isdir("data/LibriSpeech/dev-noise/"):
        os.makedirs("data/LibriSpeech/dev-noise/")
    # Loop through all the audio files
    for audioFile in audioFiles:
        # check if folders exist
        audioPath = '/'.join(audioFile.split('/')[2].split("\\")[1:-1])
        if not os.path.isdir("data/LibriSpeech/dev-noise/" + audioPath):
            os.makedirs("data/LibriSpeech/dev-noise/" + audioPath)

        # Load the audio file
        audio, sr = librosa.load(audioFile)
        noise_norm = preprocessing.normalize([noise[:len(audio)]])
        audio_norm = preprocessing.normalize([audio])
        # Calculate the alpha value
        snr = random.randint(-10, 10)
        alphaValue = alpha(snr)
        # Add noise to the audio
        noisyAudio = audio_norm + alphaValue * noise_norm
        # Save the noisy audio
        fileName = os.path.basename(audioFile)
        filePath = "data/LibriSpeech/dev-noise/" + audioPath + "/" + fileName.split(".")[0] + "_" + str(snr) + ".wav"
        sf.write(filePath, noisyAudio.T,sr)
        i += 1
        print(filePath + " Saved Successfully " + str(i/x*100) + "%")