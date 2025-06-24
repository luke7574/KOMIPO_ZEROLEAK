import os
import shutil
import torchaudio
import torch
from asteroid.models import ConvTasNet
import soundfile as sf
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.fftpack import fft

one = 0    # wav파일명으로 폴더 생성후 하위폴더로 FFT/MEL/STFT 폴더 생성
two = 1    # ConvTasNet모델 활용하여 복원2개 생성
three = 1  # FFT / MEL / STFT  그래프 생성하여 알맞는 폴더로 이동

# ✅ 상위 폴더 경로 설정
base_input_dir = "C:/Users/user/중부발전/M2_Leak/0613_0619/FH103"

def get_wav_clean1sec(signal,sr):
    SEC_0_1 = sr // 10  # 0.1초 샘플 개수
    SEC_1 = sr          # 1초 샘플 개수
    duration = int(len(signal) / sr)  # 오디오의 총 길이 (초단위)
    s_fft = []
    i_time = (duration - 1) * 10 - 1  # 검사할 1초 구간의 개수
    for i in range(i_time):
        u_data = signal[(i + 1) * SEC_0_1:(i + 1) * SEC_0_1 + SEC_1] # 100ms 간격으로 이동하며 1초 길이의 신호 추출
        s_fft.append(np.std(u_data))
    a = np.argmin(s_fft) + 1
    tfa_data = signal[a * SEC_0_1: a * SEC_0_1 + SEC_1]
    return tfa_data, sr
#---------------------------------------------------------------------------------------------------------
if one:
    # ✅ 폴더 내 모든 파일 반복
    for filename in os.listdir(base_input_dir):
        if filename.endswith(".wav"):
            wav_path = os.path.join(base_input_dir, filename)
            name_only = os.path.splitext(filename)[0]
            # ✅ 파일 이름에서 .wav 제거하고 _undefined 제거
            name_only = os.path.splitext(filename)[0].replace('_undefined', '')
            
            # ✅ 동일 이름의 하위 폴더 생성
            target_folder = os.path.join(base_input_dir, name_only)
            os.makedirs(target_folder, exist_ok=True)
            
            new_filename = filename.replace('_undefined', '')
            # ✅ WAV 파일 이동
            target_path = os.path.join(target_folder, new_filename)
            shutil.move(wav_path, target_path)
            print(f"✅ {filename} → {target_path} 로 이동 완료")
            
            # ✅ FFT, MEL, STFT 폴더 생성
            for subfolder in ["FFT", "MEL", "STFT", "테스트"]:
                subfolder_path = os.path.join(target_folder, subfolder)
                os.makedirs(subfolder_path, exist_ok=True)
                print(f"📁 {subfolder} 폴더 생성 완료 → {subfolder_path}")
                if subfolder == "테스트":
                    lowfolder = "remove_elec"
                    lowfolder_path = os.path.join(subfolder_path, lowfolder)
                    os.makedirs(lowfolder_path, exist_ok=True)
#--------------------------------------------------------------------------------------------------------
if two:
    # ✅ ConvTasNet 모델 불러오기
    model = ConvTasNet.from_pretrained("JorisCos/ConvTasNet_Libri2Mix_sepclean_16k")
    
    # ✅ 하위 폴더 순회
    for subfolder_name in os.listdir(base_input_dir):
        subfolder_path = os.path.join(base_input_dir, subfolder_name)

        if not os.path.isdir(subfolder_path):
            continue

        # 해당 하위 폴더 내의 wav 파일 검색
        wav_files = [f for f in os.listdir(subfolder_path) if f.endswith(".wav") and "_clean" not in f]
        if not wav_files:
            continue

        # 첫 번째 wav 파일 선택
        wav_file = wav_files[0]
        input_wav_path = os.path.join(subfolder_path, wav_file)
        wav_filename = os.path.splitext(wav_file)[0]

        # 오디오 로드 및 전처리
        waveform, sr = torchaudio.load(input_wav_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # ConvTasNet 분리 실행
        with torch.no_grad():
            separated_sources = model.separate(waveform)

        # 원본 오디오 길이에 맞게 자름
        original_len = waveform.shape[1]
        sources = separated_sources[0, :, :original_len].cpu().numpy()

        # 같은 폴더 내에 clean 파일 저장
        for i, data in enumerate(sources):
            data = data.astype(np.float32)
            out_path = os.path.join(subfolder_path, f"{wav_filename}_part{i+1}.wav")
            try:
                sf.write(out_path, data, sr, format='WAV', subtype='PCM_16')
                print(f"✅ 저장 완료: {out_path}")
            except Exception as e:
                print(f"❌ 저장 실패: {out_path} → {e}")

#--------------------------------------------------------------------------------------------------------
if three:
    # ✅ 3단계: 모든 wav에 대해 FFT/MEL/STFT 생성
    def generate_spectrograms(data, sr, save_prefix, folder_path):
        # FFT
        fft_data = abs(fft(data))
        half = len(fft_data) // 2
        fft_data = fft_data[:half]
        freqs = np.arange(half) * sr / len(data)
        plt.figure(figsize=(12, 4))
        plt.plot(freqs, fft_data)
        plt.title("FFT Spectrum")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        plt.savefig(os.path.join(folder_path, "FFT", f"{save_prefix}_FFT.png"))
        plt.close()

        # MEL
        S = librosa.feature.melspectrogram(y=data, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)
        plt.figure(figsize=(12, 4))
        librosa.display.specshow(S_db, sr=sr, hop_length=512, x_axis='time', y_axis='hz', cmap='magma')
        plt.colorbar(format="%+2.0f dB")
        plt.title("Mel Spectrogram")
        plt.tight_layout()
        plt.savefig(os.path.join(folder_path, "MEL", f"{save_prefix}_MEL.png"))
        plt.close()

        # STFT
        D = librosa.stft(data, n_fft=1024, hop_length=512)
        D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        plt.figure(figsize=(12, 4))
        librosa.display.specshow(D_db, sr=sr, hop_length=512, x_axis='time', y_axis='hz', cmap='magma')
        plt.colorbar(format="%+2.0f dB")
        plt.title("STFT Spectrogram")
        plt.tight_layout()
        plt.savefig(os.path.join(folder_path, "STFT", f"{save_prefix}_STFT.png"))
        plt.close()

    # ✅ 스펙트로그램 생성 적용 (librosa.load로 5초까지 로드)
    for folder in os.listdir(base_input_dir):
        folder_path = os.path.join(base_input_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        for file in os.listdir(folder_path):
            if file.endswith(".wav"):
                wav_path = os.path.join(folder_path, file)
                try:
                    # librosa로 5초까지 로드
                    data, samplerate = librosa.load(wav_path, sr=None, duration=5)
                    data, samplerate = get_wav_clean1sec(data, samplerate)
                    base = os.path.splitext(file)[0]
                    generate_spectrograms(data, samplerate, base, folder_path)
                    print(f"✅ 그래프 생성 완료: {file}")
                except Exception as e:
                    print(f"❌ 오류 발생: {file} → {e}")

