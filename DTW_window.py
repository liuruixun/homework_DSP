import os
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import re
import pyaudio
import wave
import librosa
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, scrolledtext


def short_time_mfcc(y, sr, frame_length, hop_length, n_mfcc=13):
    mfccs = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc, n_fft=frame_length, hop_length=hop_length)
    return mfccs


def compute_features(y, sr):
    frame_length = int(0.01 * sr)  # 25ms
    hop_length = int(0.005 * sr)    # 10ms

    energy = np.array([
        np.sum(np.abs(y[i:i + frame_length])**2)
        for i in range(0, len(y), hop_length)
    ])

    zero_crossings = librosa.feature.zero_crossing_rate(
        y, frame_length=frame_length, hop_length=hop_length)[0]

    rms = np.sqrt(energy)

    variance = np.array([
        np.var(y[i:i + frame_length])
        for i in range(0, len(y), hop_length)
    ])

    mean = np.array([
        np.mean(y[i:i + frame_length])
        for i in range(0, len(y), hop_length)
    ])

    max_val = np.array([
        np.max(y[i:i + frame_length])
        for i in range(0, len(y), hop_length)
    ])
    min_val = np.array([
        np.min(y[i:i + frame_length])
        for i in range(0, len(y), hop_length)
    ])

    mfcc_features = short_time_mfcc(y, sr, frame_length, hop_length)

    return energy, zero_crossings, rms, variance, mean, max_val, min_val, mfcc_features.transpose((1, 0))


def endpoint_detection(audio_file, plot_area):
    y, sr = librosa.load(audio_file, sr=None)

    energy, zero_crossings, rms, variance, mean, max_val, min_val, mfcc_features = compute_features(
        y, sr)
    energy_threshold_high = np.mean(energy[:int(0.5*sr)]) * 3
    energy_threshold_low = np.mean(energy[:int(0.5*sr)]) * 1.5

    speech_frames = []
    is_speech = False

    for i in range(len(energy)):
        if energy[i] > energy_threshold_high and not is_speech:
            is_speech = True
            start_frame = i
        elif energy[i] < energy_threshold_low and is_speech:
            is_speech = False
            end_frame = i
            speech_frames.append((start_frame, end_frame))

    frame_length = int(0.025 * sr)  # 25ms
    hop_length = int(0.01 * sr)
    times = librosa.frames_to_time(
        np.arange(len(energy)), sr=sr, hop_length=hop_length)


    plot_area.clear()
    plot_area.plot(times, energy, label='Energy')
    plot_area.plot(times, zero_crossings, label='Zero Crossing Rate')
    plot_area.axhline(y=energy_threshold_high, color='r',
                       linestyle='--', label='High Threshold')
    plot_area.axhline(y=energy_threshold_low, color='g',
                       linestyle='--', label='Low Threshold')
    plot_area.set_title('Energy and Zero Crossing Rate')
    plot_area.set_xlabel('Time (s)')
    plot_area.set_ylabel('Amplitude')
    plot_area.legend()
    for start, end in speech_frames:
        plot_area.axvspan(times[start], times[end], color='yellow', alpha=0.3)
    for start, end in speech_frames:
        if end-start < 20:
            continue
        energy, zero_crossings, rms, variance, mean, max_val, min_val, mfcc_features = energy[
            start:end], zero_crossings[start:end], rms[start:end], variance[start:end], mean[start:end], max_val[start:end], min_val[start:end], mfcc_features[start:end, :]
        return mfcc_features


def get_audio(filepath):
    CHUNK = 256
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 8000
    RECORD_SECONDS = 5
    WAVE_OUTPUT_FILENAME = filepath
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    frames = []
    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()


def calculate_dtw(signal1, signal2):
    distance, path = fastdtw(signal1, signal2, dist=euclidean)
    return distance


class AudioApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Processing App")

        self.frame = tk.Frame(root)
        self.frame.pack()

        self.record_button = tk.Button(
            self.frame, text="Record and Process", command=self.process_audio)
        self.record_button.pack()

        self.progress = ttk.Progressbar(
            self.frame, orient='horizontal', mode='determinate', length=280)
        self.progress.pack(pady=10)

        self.result_text = scrolledtext.ScrolledText(
            self.frame, width=40, height=10)
        self.result_text.pack()

        self.figure = plt.Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.frame)
        self.plot_area = self.figure.add_subplot(111)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def process_audio(self):
        in_path = 'test.wav'
        get_audio(in_path)
        mfcc_features = endpoint_detection(in_path, self.plot_area)
        self.canvas.draw()

        folder_path = 'features/train'
        train_files = [f for f in os.listdir(
            folder_path) if f.endswith('.npy')]

        data_test = mfcc_features

        min_distance = float('inf')
        closest_template = None

        dis = []
        id = []

        self.progress['maximum'] = len(train_files)

        for index, filename_train in enumerate(train_files):
            file_path_train = os.path.join(folder_path, filename_train)
            data_train = np.load(file_path_train)[:, 7:21]
            distance = calculate_dtw(data_test, data_train)
            dis.append(distance)
            id.append(filename_train)
            if distance < min_distance:
                min_distance = distance
                closest_template = filename_train

            self.progress['value'] = index + 1
            self.root.update_idletasks()

        dis = np.array(dis)
        sorted_indices = np.argsort(dis)
        top_5_indices = sorted_indices[:5]

        results = "相似度top5预测类别:\n"
        for i in top_5_indices:
            predicted_label = int(re.findall(r'\d+', id[i])[0])
            results += f"{predicted_label}\n"

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, results)


if __name__ == "__main__":
    root = tk.Tk()
    app = AudioApp(root)
    root.mainloop()
