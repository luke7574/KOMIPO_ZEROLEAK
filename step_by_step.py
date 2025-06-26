import librosa
import numpy as np
import os
from scipy.signal import find_peaks, iirnotch, filtfilt
import soundfile as sf

# 🔧 설정
SR = 8000
N_FFT = 2048
HOP_LENGTH = 512
BIN_RESOLUTION = SR / N_FFT  # ≈ 3.90625 Hz

INPUT_FOLDER = "C:/Users/user/AI/KOMIPO_ZeroLeak/step_test"
CLEAN_SEGMENT_FOLDER = "C:/Users/user/AI/KOMIPO_ZeroLeak/잡음_제거전"
OUTPUT_FOLDER = "C:/Users/user/AI/KOMIPO_ZeroLeak/잡음_제거후"

# 깨끗한 1초 선택하기
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
    sec_data = signal[a * SEC_0_1: a * SEC_0_1 + SEC_1]
    return sec_data, sr

# 깨끗한 1초 음원 복원하기
def save_clean_1sec_segment(filepath):
    y, _ = librosa.load(filepath, sr=SR)
    y_sec, _ = get_wav_clean1sec(y, SR)

    filename = os.path.basename(filepath)
    name, ext = os.path.splitext(filename)
    new_filename = f"{name}_before{ext}"
    output_path = os.path.join(CLEAN_SEGMENT_FOLDER, new_filename)

    sf.write(output_path, y_sec.astype(np.float32), SR)
    print(f"Saved clean 1-sec: {output_path}")


# 모든 음원에 대해 STFT 또는 FFT 수행하기
def compute_magnitude_spectrum(file_path, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH):
    y, _ = librosa.load(file_path, sr=sr)
    y_sec, _ = get_wav_clean1sec(y,sr)
    stft = librosa.stft(y_sec, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    return magnitude

# 모든 음원의 스펙트럼 평균화
all_mags = []

for file in os.listdir(INPUT_FOLDER):
    if file.endswith(".wav"):
        mag = compute_magnitude_spectrum(os.path.join(INPUT_FOLDER, file))
        # print(mag)
        # print(len(mag)) # 1025
        all_mags.append(mag)

# 시간 축 평균 후 주파수별 평균 값 계산 (노이즈가 일정하면 주로 피크 형성됨)
mean_spectrum = np.mean([np.mean(mag, axis=1) for mag in all_mags], axis=0)
# print(mean_spectrum)
# print(len(mean_spectrum)) # 1025

# 일정 이상 강도인 주파수 대역만 추출 (노이즈로 의심)
peaks, _ = find_peaks(mean_spectrum, height=np.percentile(mean_spectrum, 95))

# 주파수 대역 나타내기
for peak in peaks:
    freq = peak * BIN_RESOLUTION
    print(f"bin {peak} -> frequency: {freq:.2f} Hz")


# 노이즈 제거 및 복원
def remove_noise_and_save(filepath, peaks):
    y, _ = librosa.load(filepath, sr=SR)
    y_sec, _ = get_wav_clean1sec(y,SR)
    D = librosa.stft(y_sec, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mag, phase = np.abs(D), np.angle(D)
###################################################################
    # # 🔧 노이즈 bin ±5 대역 제거
    # for bin_idx in peaks:
    #     for offset in range(-5, 6):  # -5 ~ +5
    #         target_idx = bin_idx + offset
    #         if 0 <= target_idx < mag.shape[0]:
    #             mag[target_idx, :] = 0
###################################################################
    # ✅ 중심 주파수 bin만 보존 (±5 범위)
    preserve_bins = set()
    for bin_idx in peaks:
        for offset in range(-5, 6):
            target_idx = bin_idx + offset
            if 0 <= target_idx < mag.shape[0]:
                preserve_bins.add(target_idx)
    # ✅ 나머지 bin 제거
    for i in range(mag.shape[0]):
        if i not in preserve_bins:
            mag[i, :] = 0
###################################################################

    # 복원
    D_cleaned = mag * np.exp(1j * phase)
    y_cleaned = librosa.istft(D_cleaned, hop_length=HOP_LENGTH)

    # 저장
    filename = os.path.basename(filepath)
    name, ext = os.path.splitext(filename)
    new_filename = f"{name}_remove{ext}"
    output_path = os.path.join(OUTPUT_FOLDER, new_filename)
    sf.write(output_path, y_cleaned.astype(np.float32), SR)
    print(f"Saved: {output_path}")

# 🔄 Step 4: 전체 파일 처리
for file in os.listdir(INPUT_FOLDER):
    if file.endswith(".wav"):
        full_path = os.path.join(INPUT_FOLDER, file)
        save_clean_1sec_segment(full_path)
        remove_noise_and_save(full_path, peaks)