import pandas as pd
import numpy as np
import librosa
import librosa.display
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import soundfile as sf
import os

made_csv = 0
melspectrogram = 1
stft_spectrogram = 1
fft_img = 1
stft_wav = 0

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

# ✅ 처리할 디렉토리 지정
input_folder = "C:/Users/user/AI/KOMIPO_ZeroLeak/test/convtasnet_test_clean/190228_20250611_10_27_09_126_N"
wav_files = [f for f in os.listdir(input_folder) if f.endswith(".wav")]

for wav_file in wav_files:
    wav_path = os.path.join(input_folder, wav_file)
    print(f"▶ 처리 중: {wav_path}")

    data, samplerate = librosa.load(wav_path, sr=None, duration=5)

    wav_filename = os.path.splitext(os.path.basename(wav_path))[0]
    base_output_dir = os.path.dirname(wav_path)
    plot_path = os.path.join(base_output_dir, wav_filename)
    os.makedirs(plot_path, exist_ok=True)

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
        df.to_csv(save_folder, index=False)
        print(f"결과가 CSV로 저장되었습니다: {save_folder}")

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


    if stft_wav:
        save_wav_path = "C:/Users/user/AI/KOMIPO_ZeroLeak/test/가나다/cleaned_filtered_output.wav"
        low_threshold_hz = 1600  # 이 이상의 주파수를 노이즈로 간주
        high_threshold_hz = 1700
        # 2. STFT 변환
        D = librosa.stft(data, n_fft=1024, hop_length=512)
        D_mag = np.abs(D)
        D_phase = np.angle(D)

        # 3. 주파수 벡터 생성
        freqs = librosa.fft_frequencies(sr=samplerate, n_fft=1024)  # 길이: 513
        freq_mask = (freqs > low_threshold_hz) & (freqs < high_threshold_hz)              # 노이즈 범위 마스크 (True인 부분 제거)

        # 4. 노이즈 제거: 주파수 성분을 0으로 마스킹
        D_mag_cleaned = D_mag.copy()
        D_mag_cleaned[~freq_mask, :] = 0

        # 5. 복소수로 다시 합성 (magnitude + phase)
        D_cleaned_complex = D_mag_cleaned * np.exp(1j * D_phase)

        # 6. iSTFT를 통해 시간 도메인 신호로 복원
        y_cleaned = librosa.istft(D_cleaned_complex, hop_length=512)

        # 7. WAV 저장 (float32)
        sf.write(save_wav_path, y_cleaned.astype(np.float32), samplerate)
        print(f"✅ 노이즈 제거 후 WAV 저장 완료: {save_wav_path}")

        # 8. 시각화 (비교용)
        plt.figure(figsize=(12, 5))
        librosa.display.specshow(librosa.amplitude_to_db(D_mag_cleaned, ref=np.max), sr=samplerate, hop_length=512,
                                x_axis='time', y_axis='hz', cmap='magma')
        plt.colorbar(format='%+2.0f dB')
        plt.title("STFT (After Noise Filtering > 3kHz)")
        plt.tight_layout()
        plt.savefig("C:/Users/user/AI/KOMIPO_ZeroLeak/test/가나다/stft_after.png")
        # plt.show()





