import numpy as np
import numpy.fft as fft
import sys
import librosa
from scipy import *
import os




def hps(signal, steps, fft_length):
    f_signal = fft.fft(signal, fft_length)
    f_signal = f_signal[:len(f_signal) // 2]
    f_signal = np.abs(f_signal)

    result_length = len(f_signal) // steps
    result = f_signal[:result_length].copy()

    for i in range(2, steps + 1):
        result = result * f_signal[::i][:result_length]

    return result


def divide(signal, piece_length, overlap_length):
    d = piece_length - overlap_length
    pieces = []
    for i in range(0, len(signal) - piece_length + 1, d):
        pieces.append(signal[i:i + piece_length])
    return pieces


def window(pieces):
    windowed_pieces = []
    for ch in pieces:
        windowed_pieces.append(ch * np.hamming(len(ch)))
    return windowed_pieces


def find_fundamental_frequency(hps_result, sample_rate, signal_length):
    min_f = 85
    start_i = int((min_f / sample_rate) * signal_length)

    mx_i = start_i
    for i in range(start_i + 1, len(hps_result)):
        if hps_result[i] > hps_result[mx_i]:
            mx_i = i

    return (mx_i / signal_length) * sample_rate


def recognize_gender(file):
    signal, w = librosa.load(file)

    signal = signal.astype(float) / 2 ** 16
    w = float(w)

    piece_length = 16 * 1024
    overlap_length = piece_length // 2
    hps_steps = 4
    fft_length = 4 * piece_length

    pieces = divide(signal, piece_length, overlap_length)
    pieces = window(pieces)

    frequencies = []
    for p in pieces:
        hps_result = hps(p, hps_steps, fft_length)
        frequencies.append(find_fundamental_frequency(hps_result, w, fft_length))
    fr = np.median(frequencies)
    if fr < 165:
        return 'M'
    else:
        return 'K'


def main():
    files = sys.argv[1:]
    for i in files:
        recognize_gender(i)

    if (len(files) > 0):
        return

    countFiles = 0
    correctRecognition = 0
    for filename in os.listdir('train'):
        filePath = os.path.join('train', filename)

        if filename.endswith(".wav"):


            countFiles = countFiles + 1
            recognition = recognize_gender(filePath)

            if (recognition == filename[4]):
                correctRecognition = correctRecognition + 1
            #else:
                #print("This file is not correctly recognized: ",filePath)

            print(filePath, recognition)

    print('correctly recognized:', correctRecognition, 'All:', countFiles, 'percentage recognized:',correctRecognition / countFiles * 100)


if __name__ == '__main__':
    main()
