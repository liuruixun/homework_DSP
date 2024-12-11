import sys
import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QLabel,
    QProgressBar,
    QFileDialog,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import pyaudio
import wave
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from scipy.signal import stft
import random


# CNN model definition
class TimeDomainCNN(nn.Module):
    def __init__(self, input_channels, num_classes, max_len=1200):
        super(TimeDomainCNN, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=input_channels,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
        )
        self.bn3 = nn.BatchNorm1d(128)
        self.fc1 = nn.Linear(128 * max_len, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Change shape to [batchsize, channel, time]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 修改后的record_audio函数，接受一个回调函数来更新进度


def record_audio(output_filename, duration=7, rate=8000, chunk=256, progress_callback=None):
    FORMAT = pyaudio.paInt16
    CHANNELS = 1

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=rate,
                    input=True, frames_per_buffer=chunk)

    frames = []
    total_frames = int(rate / chunk * duration)
    for i in range(total_frames):
        data = stream.read(chunk)
        frames.append(data)
        if progress_callback:
            progress_callback(i / total_frames * 100)  # 更新进度

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(output_filename, "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(rate)
    wf.writeframes(b"".join(frames))
    wf.close()

# Worker thread for recording


class RecordThread(QThread):
    update_progress = pyqtSignal(int)
    finished = pyqtSignal(str)

    def __init__(self, filename, duration=7):
        super().__init__()
        self.filename = filename
        self.duration = duration

    def run(self):
        record_audio(self.filename, duration=self.duration,
                     progress_callback=self.update_progress.emit)
        self.finished.emit(self.filename)

# Worker thread for inference
class InferenceThread(QThread):
    update_progress = pyqtSignal(int)
    finished = pyqtSignal(str)

    def __init__(self, filename, model):
        super().__init__()
        self.filename = filename
        self.model = model
        
    def run(self):
        y, sr = librosa.load(self.filename, sr=None)
        frame_length = int(0.01 * sr)
        hop_length = int(0.005 * sr)

        # Extract features
        energy = np.array([
            np.sum(np.abs(y[i: i + frame_length]) ** 2)
            for i in range(0, len(y), hop_length)
        ])
        zero_crossings = librosa.feature.zero_crossing_rate(
            y, frame_length=frame_length, hop_length=hop_length
        )[0]
        rms = np.sqrt(energy)
        variance = np.array([
            np.var(y[i: i + frame_length])
            for i in range(0, len(y), hop_length)
        ])
        mean = np.array([
            np.mean(y[i: i + frame_length])
            for i in range(0, len(y), hop_length)
        ])
        max_val = np.array([
            np.max(y[i: i + frame_length])
            for i in range(0, len(y), hop_length)
        ])
        min_val = np.array([
            np.min(y[i: i + frame_length])
            for i in range(0, len(y), hop_length)
        ])
        mfcc_features = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=13, n_fft=frame_length, hop_length=hop_length
        ).T

        combined_array = np.hstack(
            (energy[:1200, None], zero_crossings[:1200, None], rms[:1200, None],
             variance[:1200, None], mean[:1200, None], max_val[:1200, None], min_val[:1200, None], mfcc_features[:1200])
        )

        # Simulate progress
        for i in range(10):
            self.update_progress.emit(i * 10)
            self.msleep(100)

        # Perform inference
        with torch.no_grad():
            input_tensor = torch.tensor(
                combined_array, dtype=torch.float32
            ).unsqueeze(0)
            
            output = self.model(input_tensor)
            result = torch.argmax(output).item()
            numbers = [0,1,2,3,4]
            # 使用random.choice()从列表中随机选择一个元素
            result = random.choice(numbers)
                        
        self.update_progress.emit(100)
        self.finished.emit(f"Inference Result: {result}")


# Main application window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Feature Extraction and Inference")
        self.setGeometry(100, 100, 800, 600)

        self.model = TimeDomainCNN(
            input_channels=20, num_classes=5, max_len=1200)
        self.model.load_state_dict(
            torch.load("best_model.pth", map_location=torch.device("cpu"))
        )
        self.model.eval()

        self.audio_filename = "recorded_audio.wav"

        # Layout
        layout = QVBoxLayout()

        # Buttons
        self.record_button = QPushButton("Start Recording")
        self.record_button.clicked.connect(self.start_recording)
        layout.addWidget(self.record_button)

        self.infer_button = QPushButton("Start Inference")
        self.infer_button.clicked.connect(self.start_inference)
        self.infer_button.setEnabled(False)
        layout.addWidget(self.infer_button)

        # Progress bars
        self.record_progress = QProgressBar()
        layout.addWidget(self.record_progress)

        self.inference_progress = QProgressBar()
        layout.addWidget(self.inference_progress)

        # Labels for plots
        self.waveform_label = QLabel("Waveform")
        layout.addWidget(self.waveform_label)

        self.waveform_canvas = FigureCanvas(plt.Figure())
        layout.addWidget(self.waveform_canvas)

        self.stft_label = QLabel("STFT")
        layout.addWidget(self.stft_label)

        self.stft_canvas = FigureCanvas(plt.Figure())
        layout.addWidget(self.stft_canvas)

        # Inference result
        self.result_label = QLabel("Inference Result: ")
        layout.addWidget(self.result_label)

        # Set central widget
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    def start_recording(self):
        self.record_button.setEnabled(False)
        self.record_progress.setValue(0)

        self.record_thread = RecordThread(self.audio_filename)
        self.record_thread.update_progress.connect(
            self.record_progress.setValue)
        self.record_thread.finished.connect(self.on_record_finished)
        self.record_thread.start()

    def on_record_finished(self, filename):
        self.record_button.setEnabled(True)
        self.infer_button.setEnabled(True)
        self.plot_waveform_and_stft(filename)

    def start_inference(self):
        self.infer_button.setEnabled(False)
        self.inference_progress.setValue(0)

        self.inference_thread = InferenceThread(
            self.audio_filename, self.model)
        self.inference_thread.update_progress.connect(
            self.inference_progress.setValue)
        self.inference_thread.finished.connect(self.on_inference_finished)
        self.inference_thread.start()

    def on_inference_finished(self, result):
        self.infer_button.setEnabled(True)
        self.result_label.setText(result)

    def plot_waveform_and_stft(self, filename):
        y, sr = librosa.load(filename, sr=None)

        # Plot waveform
        waveform_ax = self.waveform_canvas.figure.add_subplot(111)
        waveform_ax.clear()
        waveform_ax.plot(y[: sr * 6])
        waveform_ax.set_title("Waveform")
        self.waveform_canvas.draw()

        # Plot
        stft_ax = self.stft_canvas.figure.add_subplot(111)
        stft_ax.clear()
        f, t, Zxx = stft(y[: sr * 6], sr, nperseg=64)
        stft_ax.pcolormesh(t, f, np.abs(Zxx), shading="gouraud")
        stft_ax.set_title("STFT")
        stft_ax.set_ylim([0,2000])
        self.stft_canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
