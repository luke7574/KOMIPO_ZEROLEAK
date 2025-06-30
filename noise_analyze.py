import pandas as pd
import numpy as np
import librosa
import librosa.display
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import soundfile as sf
import os

made_csv = 0
melspectrogram = 0
stft_spectrogram = 0
fft_img = 0
stft_wav = 0
remove_electronic = 1

# plot_path = "C:/Users/user/AI/KOMIPO_ZeroLeak/test/convtasnet_test_clean/153606_20231010_09_11_01_127_N"


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

# wav_path = "C:/Users/user/중부발전/M2_Leak/0613_0619/FH103/3026_20250619_053000/3026_20250619_053000.wav"
wav_path = "/Users/wook/WIPLAT/중부발전/M2_Leak/0613_0619/V110/2940_20250613_033000/2940_20250613_033000.wav"  # MacBook
data, samplerate = librosa.load(wav_path, sr=None, duration=5)

# wav_filename = os.path.splitext(os.path.basename(wav_path))[0]
# base_output_dir = os.path.dirname(wav_path)
# plot_path = os.path.join(base_output_dir, wav_filename)
# os.makedirs(plot_path, exist_ok=True)


#----------------------------------------------------------------------------------------------------------
if made_csv:
    fft_data = abs(fft(data))
    # 전체 fft_data에서 최대 진폭 주파수 계산
    max_index = np.argmax(fft_data)
    hz_per_bin = samplerate / len(fft_data)  # 1 bin당 주파수 간격
    max_freq = max_index * hz_per_bin

    print(f"전체 구간 max FFT 값: {fft_data[max_index]:.2f}")
    print(f"해당 주파수(Hz): {max_freq:.2f} Hz")
    #----------------------------------------------------------------------------------------------
    data_1sec, sr = get_wav_clean1sec(data,samplerate)
    fft_data_1sec = abs(fft(data_1sec))

    max_index_1sec = np.argmax(fft_data_1sec)
    hz_per_bin_1sec = samplerate / len(fft_data_1sec)
    max_freq_1sec = max_index_1sec * hz_per_bin_1sec

    print(f"1초 구간 max FFT 값: {fft_data_1sec[max_index_1sec]:.2f}")
    print(f"해당 주파수(Hz): {max_freq_1sec:.2f} Hz")

    df = pd.DataFrame({
        'HZ' : np.arange(len(fft_data))*hz_per_bin,
        'fft' : fft_data
    })
    save_folder = "C:/Users/user/AI/KOMIPO_ZeroLeak/electric_sound/fft_data_abs.csv"
    save_folder_mac = "/Users/wook/WIPLAT/중부발전/M2_Leak/0613_0619/V110/2940_20250613_033000/fft_data_abs.csv"
    df.to_csv(save_folder_mac, index=False)
    print(f"결과가 CSV로 저장되었습니다: {save_folder_mac}")

#----------------------------------------------------------------------------------------------------------
if melspectrogram:
    # 2. Mel 스펙트로그램 계산
    S = librosa.feature.melspectrogram(y=data, sr=samplerate, n_fft=2048, hop_length=512, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)  # dB scale로 변환
    # 3. 시각화
    plt.figure(figsize=(12, 5))
    librosa.display.specshow(S_db, sr=samplerate, hop_length=512,
                            x_axis='time', y_axis='hz', cmap='magma')

    plt.colorbar(format="%+2.0f dB")
    plt.title("Mel Spectrogram (Time vs Frequency)")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.tight_layout()
    out_path = os.path.join(plot_path, "mel.png")
    plt.savefig(out_path)
    # plt.show()
    

# 🔹 STFT 스펙트로그램 시각화
if stft_spectrogram:
    D = librosa.stft(data, n_fft=1024, hop_length=512)
    D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    # # 시간 및 주파수 축 계산
    # freqs = librosa.fft_frequencies(sr=samplerate, n_fft=1024)   # 주파수 벡터 (shape: 513,)
    # times = librosa.frames_to_time(np.arange(D_db.shape[1]), sr=samplerate, hop_length=512)  # 시간 벡터

    # # DataFrame으로 변환 (행: 주파수, 열: 시간)
    # df_stft = pd.DataFrame(D_db, index=freqs, columns=times)
    # df_stft.index.name = "Frequency (Hz)"
    # df_stft.columns.name = "Time (s)"

    # # CSV 저장 경로
    # stft_csv_path = "C:/Users/user/AI/KOMIPO_ZeroLeak/test/가나다/stft_spectrogram.csv"
    # df_stft.to_csv(stft_csv_path)
    # print(f"✅ STFT dB 데이터가 CSV로 저장되었습니다: {stft_csv_path}")

    plt.figure(figsize=(12, 5))
    librosa.display.specshow(D_db, sr=samplerate, hop_length=512,
                              x_axis='time', y_axis='hz', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title('STFT Spectrogram (Time vs Frequency)')
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.tight_layout()
    out_path = os.path.join(plot_path, "stft.png")
    plt.savefig(out_path)
    # plt.show()
    
if fft_img:
    fft_data = abs(fft(data))
    # Nyquist 주파수까지만 사용
    half_len = len(fft_data) // 2
    fft_data = fft_data[:half_len]
    hz_per_bin = samplerate / len(fft_data) / 2  # bin당 주파수 간격
    freqs = np.arange(half_len) * samplerate / len(fft_data) / 2
    # 최대 진폭 주파수
    max_index = np.argmax(fft_data)
    max_freq = freqs[max_index]

    print(f"전체 구간 max FFT 값: {fft_data[max_index]:.2f}")
    print(f"해당 주파수(Hz): {max_freq:.2f} Hz")

    # ✅ FFT 그래프 시각화 및 저장
    plt.figure(figsize=(12, 5))
    freqs = np.arange(len(fft_data)) * hz_per_bin
    plt.plot(freqs, fft_data, label='FFT Spectrum')
    # plt.axvline(max_freq, color='r', linestyle='--', label=f'Max: {max_freq:.1f}Hz')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.title("FFT Spectrum")
    plt.legend()
    # plt.grid(True)

    out_path = os.path.join(plot_path, "fft.png")
    plt.savefig(out_path)
    plt.close()
    print(f"✅ FFT 그래프 저장 완료: {out_path}")

# ✅ FFT 이미지 저장 함수
def save_fft_plot(signal, samplerate, title, save_path):
    fft_data = np.abs(fft(signal))
    half_len = len(fft_data) // 2
    fft_data = fft_data[:half_len]
    freqs = np.linspace(0, samplerate / 2, half_len)

    plt.figure(figsize=(12, 5))
    plt.plot(freqs, fft_data)
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✅ FFT 그래프 저장 완료: {save_path}")
# ✅ STFT 이미지 저장 함수
def save_stft_plot(signal, samplerate, title, save_path):
    plt.figure(figsize=(12, 5))
    librosa.display.specshow(librosa.amplitude_to_db(signal, ref=np.max), sr=samplerate, hop_length=512,
                            x_axis='time', y_axis='hz', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✅ FFT 그래프 저장 완료: {save_path}")
# ✅ MEL 이미지 저장 함수
def save_mel_plot(signal, samplerate, title, save_path):
    S = librosa.feature.melspectrogram(y=signal, sr=samplerate, n_fft=2048, hop_length=512, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)
    
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(S_db, sr=samplerate, hop_length=512, x_axis='time', y_axis='hz', cmap='magma')
    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✅ MEL 스펙트로그램 저장 완료: {save_path}")


if stft_wav:
    #----------------------------------------------------------------------------------------------------------
    low_threshold_hz = 155  # 이 이상의 주파수를 노이즈로 간주
    high_threshold_hz = 365
    #----------------------------------------------------------------------------------------------------------
    # 여러 주파수 대역을 노이즈로 간주
    noise_ranges = [(115, 125), (235, 245), (295, 310), (355, 365)]
    #----------------------------------------------------------------------------------------------------------

    # save_wav_path = f"C:/Users/user/중부발전/M2_Leak/0613_0619/FH103/3026_20250619_053000/테스트/output_{low_threshold_hz}_{high_threshold_hz}.wav"
    # save_wav_path_re = f"C:/Users/user/중부발전/M2_Leak/0613_0619/FH103/3026_20250619_053000/테스트/remove_{low_threshold_hz}_{high_threshold_hz}.wav"
    save_wav_path = f"/Users/wook/WIPLAT/중부발전/M2_Leak/0613_0619/V110/2940_20250613_033000/테스트/output_{low_threshold_hz}_{high_threshold_hz}.wav"        # MacBook
    save_wav_path_re = f"/Users/wook/WIPLAT/중부발전/M2_Leak/0613_0619/V110/2940_20250613_033000/테스트/remove_{low_threshold_hz}_{high_threshold_hz}.wav"     # MacBook
    # 2. STFT 변환
    D = librosa.stft(data, n_fft=1024, hop_length=512)
    D_mag = np.abs(D)
    D_phase = np.angle(D)

    # 3. 주파수 벡터 생성
    freqs = librosa.fft_frequencies(sr=samplerate, n_fft=1024)  # 길이: 513
    #----------------------------------------------------------------------------------------------------------
    # freq_mask = (freqs > low_threshold_hz) & (freqs < high_threshold_hz)              # 노이즈 범위 마스크 (True인 부분 제거)
    #----------------------------------------------------------------------------------------------------------
    # 여러 구간에 해당하는 마스크 생성
    freq_mask = np.full(freqs.shape, False)
    for low, high in noise_ranges:
        freq_mask |= (freqs > low) & (freqs < high)

    #----------------------------------------------------------------------------------------------------------
    # 4. 노이즈 제거: 주파수 성분을 0으로 마스킹
    # 선택한 구간만 추출
    D_mag_cleaned = D_mag.copy()
    D_mag_cleaned[~freq_mask, :] = 0
    # 선택한 구간 제외 후 추출
    D_mag_cleaned_re = D_mag.copy()
    D_mag_cleaned_re[freq_mask, :] = 0

    # 5. 복소수로 다시 합성 (magnitude + phase)
    D_cleaned_complex = D_mag_cleaned * np.exp(1j * D_phase)
    D_cleaned_complex_re = D_mag_cleaned_re * np.exp(1j * D_phase)
    # 6. iSTFT를 통해 시간 도메인 신호로 복원
    y_cleaned = librosa.istft(D_cleaned_complex, hop_length=512)
    y_cleaned_re = librosa.istft(D_cleaned_complex_re, hop_length=512)
    # 7. WAV 저장 (float32)
    sf.write(save_wav_path, y_cleaned.astype(np.float32), samplerate)
    print(f"✅ 노이즈 제거 후 WAV 저장 완료: {save_wav_path}")
    sf.write(save_wav_path_re, y_cleaned_re.astype(np.float32), samplerate)
    print(f"✅ 노이즈 제거 후 WAV 저장 완료: {save_wav_path_re}")

    # FFT 시각화
    # ✅ 파일 경로 지정 (이미 있는 주파수 범위 활용)
    # path = f"C:/Users/user/중부발전/M2_Leak/0613_0619/FH103/3026_20250619_053000/테스트"
    path = f"/Users/wook/WIPLAT/중부발전/M2_Leak/0613_0619/V110/2940_20250613_033000/테스트"   # MacBook
    y_cleaned_1esc, samplerate = get_wav_clean1sec(y_cleaned, samplerate)
    y_cleaned_re_1esc, samplerate = get_wav_clean1sec(y_cleaned_re, samplerate)
    
    # FFT 시각화
    # ✅ FFT 이미지 저장
    fft_img_path_1 = os.path.join(path, f"FFT_output_{low_threshold_hz}_{high_threshold_hz}.png")
    fft_img_path_2 = os.path.join(path, f"FFT_remove_{low_threshold_hz}_{high_threshold_hz}.png")
    save_fft_plot(y_cleaned_1esc, samplerate, f"FFT Spectrum (Only {low_threshold_hz}–{high_threshold_hz} Hz)", fft_img_path_1)
    save_fft_plot(y_cleaned_re_1esc, samplerate, f"FFT Spectrum (Removed {low_threshold_hz}–{high_threshold_hz} Hz)", fft_img_path_2)

    # STFT 시각화
    # ✅ STFT 이미지 저장 경로
    stft_img_path_1 = os.path.join(path, f"STFT_output_{low_threshold_hz}_{high_threshold_hz}.png")
    stft_img_path_2 = os.path.join(path, f"STFT_remove_{low_threshold_hz}_{high_threshold_hz}.png")
    # ✅ STFT 계산 및 시각화 (1초 구간만)
    D_stft_1 = librosa.stft(y_cleaned_1esc, n_fft=1024, hop_length=512)
    D_stft_2 = librosa.stft(y_cleaned_re_1esc, n_fft=1024, hop_length=512)
    save_stft_plot(D_stft_1, samplerate, f"STFT Spectrum (Only {low_threshold_hz}–{high_threshold_hz} Hz)", stft_img_path_1)
    save_stft_plot(D_stft_2, samplerate, f"STFT Spectrum (Removed {low_threshold_hz}–{high_threshold_hz} Hz)", stft_img_path_2)

    # MEL 시각화
    mel_img_path_1 = os.path.join(path, f"MEL_output_{low_threshold_hz}_{high_threshold_hz}.png")
    mel_img_path_2 = os.path.join(path, f"MEL_remove_{low_threshold_hz}_{high_threshold_hz}.png")
    save_mel_plot(y_cleaned_1esc, samplerate, f"Mel Spectrogram (Only {low_threshold_hz}–{high_threshold_hz} Hz)", mel_img_path_1)
    save_mel_plot(y_cleaned_re_1esc, samplerate, f"Mel Spectrogram (Removed {low_threshold_hz}–{high_threshold_hz} Hz)", mel_img_path_2)

def adjust_spectral_peaks_with_window(y, sr, window_size=50, threshold_ratio=3, method='mean'):
    """
    Adjust spectral peaks using a sliding window approach.

    :param y: Time-domain audio signal.
    :param sr: Sampling rate of the audio.
    :param window_size: Size of the sliding window used to calculate the average.
    :param threshold_ratio: Ratio above which a point is considered a peak.
    :return: Adjusted time-domain audio signal and sampling rate.
    """
    # Perform STFT
    D = librosa.stft(y)
    D_magnitude, D_phase = librosa.magphase(D) # D_magnitude: 복소수 STFT의 크기 (진폭) / D_phase: 위상 (복원 시 필요)

    # Calculate frequency bins
    freqs = librosa.fft_frequencies(sr=sr, n_fft=D.shape[0])

    # Limit frequency range to 20-1000 Hz
    idx = np.where((freqs >= 20) & (freqs <= 1000))[0]
    limited_magnitude = D_magnitude[idx, :]

    # Calculate the overall median amplitude within the 20-1000 Hz range
    global_median = np.median(limited_magnitude) # 중앙값
    max_peak = np.max(limited_magnitude)         # 최대값

    # Only adjust peaks if the maximum peak is more than ten times the global median
    if max_peak > 10 * global_median:
        # Adjust the magnitude of peaks
        half_window = window_size // 2
        for t in range(D_magnitude.shape[1]): # 시간 프레임
            for i in range(D_magnitude.shape[0]): # 주파수
                # Define window boundaries
                start_index = max(i - half_window, 0)
                end_index = min(i + half_window + 1, D_magnitude.shape[0])
                # Compute the average or median magnitude within the window
                if method == 'median':
                    window_stat = np.median(D_magnitude[start_index:end_index, t])
                elif method == 'mean':
                    window_stat = np.mean(D_magnitude[start_index:end_index, t])

                # Check if the current point is a significant peak
                if D_magnitude[i, t] > threshold_ratio * window_stat:
                    D_magnitude[i, t] = window_stat

    # Reconstruct the STFT matrix
    adjusted_D = D_magnitude * D_phase

    # Perform the inverse STFT to convert back to time domain
    adjusted_y = librosa.istft(adjusted_D)

    return adjusted_y, sr

if remove_electronic:
    data, samplerate = get_wav_clean1sec(data, samplerate)
    adjusted_data, samplerate = adjust_spectral_peaks_with_window(data, samplerate)

    # ✅ 파일 경로 지정 (이미 있는 주파수 범위 활용)
    # path = f"C:/Users/user/중부발전/M2_Leak/0613_0619/FH103/3026_20250619_053000/테스트/remove_elec"
    path = f"/Users/wook/WIPLAT/중부발전/M2_Leak/0613_0619/V110/2940_20250613_033000/테스트/remove_elec"       # Macbook
    # 음원 복원하기
    output_wav_path = os.path.join(path, "adjusted_output.wav")
    sf.write(output_wav_path, adjusted_data.astype(np.float32), samplerate)
    print(f"✅ 복원된 음원 저장 완료: {output_wav_path}")

    # FFT 시각화
    fft_img_path = os.path.join(path, f"FFT_elec_remove.png")
    save_fft_plot(adjusted_data, samplerate, "FFT Spectrum After Spectral Peak Adjustment", fft_img_path)

