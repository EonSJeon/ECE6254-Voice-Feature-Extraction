import tkinter as tk
import threading
import pyaudio
import wave
import numpy as np
import librosa
import tensorflow as tf
import os
from scipy.signal import butter, lfilter

######################################
# Placeholder for lowpass filtering
######################################
def lowpass_filter(data, sr, cutoff=5000, order=5):
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, data)

def extract_features(file_path, h=12):
    """
    오디오 파일을 로드한 후 lowpass filtering을 적용하고,
    원본 MFCC를 추출합니다.
    """
    try:
        audio, sr = librosa.load(file_path, sr=48000, mono=True)
        audio = np.ravel(audio)  # 1차원 배열 보장
        audio_filt = lowpass_filter(audio, sr, cutoff=5000)
        
        win_length = int(0.025 * sr)
        hop_length = int(0.01 * sr)
        n_fft = win_length
        
        # 원본 MFCC 추출
        mfcc_orig = librosa.feature.mfcc(
            y=audio_filt,
            sr=sr,
            n_mfcc=h,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window='hann'
        )
        return mfcc_orig
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return None

######################################
# Global settings and variables
######################################
CHUNK = 1024           # 버퍼당 프레임 수
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100           # 녹음 샘플링 속도 (44100)
RECORD_SECONDS = 10    # (Optional) 최대 녹음 길이

is_recording = False
frames = []            # 녹음된 오디오 프레임 저장

# Accent label mapping
accent_mapping = {0: "canada", 1: "england", 2: "indian", 3: "scotland"}

# 모델 로드 (model_mfcc.h5가 동일 디렉토리 혹은 올바른 경로에 있어야 함)
model = tf.keras.models.load_model("model_70.h5")
# 녹음 저장할 임시 파일 이름
tmp_wav_path = "temp_recording.wav"

######################################
# Functions for Recording, Processing, Replay
######################################
def record_audio():
    """녹음 중지 신호가 올 때까지 마이크로부터 오디오 캡처"""
    global is_recording, frames
    p = pyaudio.PyAudio()
    
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    
    frames = []
    while is_recording:
        data = stream.read(CHUNK)
        frames.append(data)
    
    stream.stop_stream()
    stream.close()
    p.terminate()

def start_recording():
    """별도 스레드에서 녹음 시작"""
    global is_recording
    if not is_recording:
        is_recording = True
        status_label.config(text="Status: Recording...")
        threading.Thread(target=record_audio).start()

def stop_recording():
    """녹음 중지 후, 처리 및 예측 실행"""
    global is_recording
    if is_recording:
        is_recording = False
        status_label.config(text="Status: Processing...")
        root.after(100, process_and_predict)  # 녹음 스레드 종료 시간 대기

def process_and_predict():
    """녹음된 오디오를 저장하고, lowpass filtering을 포함한 MFCC 추출 후 모델에 입력하여 결과(4개 클래스의 확률)를 표시"""
    global frames, tmp_wav_path
    
    # 1) 녹음 데이터를 임시 WAV 파일에 저장
    save_wav(tmp_wav_path, frames)
    
    # 2) extract_features_with_augmentation() 함수를 이용하여 MFCC 추출 (augmentation 없이 원본만 추출)
    mfcc = extract_features(tmp_wav_path, h=12)

    if mfcc is None or len(mfcc) == 0:
        status_label.config(text="Status: Feature Extraction Error")
        return
    
    # print(mfcc.shape)
    
    # 3) 모델 입력 크기 (12, 380)를 맞추기 위해 시간 축 프레임 수를 380으로 조정
    mfcc = mfcc[:, :380]    # 380 프레임보다 길면 자름
    if mfcc.shape[1] < 380:
        pad_width = 380 - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    
    # 4) 모델 입력 형태로 리쉐이프: (1, 12, 380, 1)
    mfcc = np.expand_dims(mfcc, axis=-1)  # -> (12, 380, 1)
    mfcc = np.expand_dims(mfcc, axis=0)    # -> (1, 12, 380, 1)

    mean = -25.147128970519486
    std = 145.0552327372998

    mfcc = (mfcc - mean) / std
    
    # 5) 모델 예측 (각 클래스 확률 벡터 반환)
    prediction = model.predict(mfcc)[0]  # 예: shape (4,)
    
    # 6) 각 클래스 확률을 백분율로 표시
    result_text = (
        f"Canada: {prediction[0]*100:.1f}%\n"
        f"England: {prediction[1]*100:.1f}%\n"
        f"Indian: {prediction[2]*100:.1f}%\n"
        f"Scotland: {prediction[3]*100:.1f}%"
    )
    
    # 7) GUI에 결과 업데이트
    result_label.config(text=result_text)
    status_label.config(text="Status: Idle")

    
    # 7) GUI에 결과 업데이트
    result_label.config(text=result_text)
    status_label.config(text="Status: Idle")

    # tmp_wav_path = "temp_recording.wav"

def save_wav(filename, audio_frames):
    """녹음된 PyAudio 프레임을 WAV 파일로 저장"""
    p = pyaudio.PyAudio()
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(audio_frames))
    wf.close()

def replay_audio():
    """마지막 녹음 파일이 있으면 재생"""
    global tmp_wav_path
    if not os.path.exists(tmp_wav_path):
        status_label.config(text="Status: No recording available for replay")
        return

    status_label.config(text="Status: Replaying...")
    p = pyaudio.PyAudio()
    wf = wave.open(tmp_wav_path, 'rb')

    def play_stream():
        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True)
        data = wf.readframes(CHUNK)
        while data:
            stream.write(data)
            data = wf.readframes(CHUNK)
        stream.stop_stream()
        stream.close()
        wf.close()
        p.terminate()
        status_label.config(text="Status: Idle")

    threading.Thread(target=play_stream).start()

######################################
# Tkinter GUI
######################################
root = tk.Tk()
root.title("Audio Classification")

# Start Recording 버튼
start_btn = tk.Button(root, text="Start Recording", command=start_recording)
start_btn.pack(pady=5)

# Stop Recording 버튼
stop_btn = tk.Button(root, text="Stop Recording", command=stop_recording)
stop_btn.pack(pady=5)

# Replay 버튼
replay_btn = tk.Button(root, text="Replay Recording", command=replay_audio)
replay_btn.pack(pady=5)

# Status label
status_label = tk.Label(root, text="Status: Idle", fg="blue")
status_label.pack(pady=5)

# Result label
result_label = tk.Label(root, text="Prediction: None", fg="red", font=("Arial", 14))
result_label.pack(pady=5)

root.mainloop()
