import numpy as np
import librosa
import soundfile as sf
import os
import noisereduce as nr



leak_wav = "C:/Users/user/AI/KOMIPO_ZeroLeak/test/device_noise/강아지.wav"
noise_wav = "C:/Users/user/AI/KOMIPO_ZeroLeak/test/device_noise/device_noise.wav"
leak_cleaned_path = "C:/Users/user/AI/KOMIPO_ZeroLeak/test/device_noise"


# WAV 파일 로드
leak_data, sr = librosa.load(leak_wav, sr=None)
noise_data, _ = librosa.load(noise_wav, sr=sr)
# 길이 맞추기 (padding)
max_len = max(len(leak_data), len(noise_data))
leak_data = np.pad(leak_data, (0, max_len - len(leak_data)))
noise_data = np.pad(noise_data, (0, max_len - len(noise_data)))

#----------------------------------------------------------------------------------------------
# # STFT
# leak_stft = librosa.stft(leak_data, n_fft=2048, hop_length=512)
# noise_stft = librosa.stft(noise_data, n_fft=2048, hop_length=512)

# # 복소수 STFT 전체 차감
# clean_stft = leak_stft - noise_stft

# # iSTFT
# clean_data = librosa.istft(clean_stft, hop_length=512)

# # RMS 정규화
# rms = np.sqrt(np.mean(clean_data**2))
# clean_data = clean_data / (rms + 1e-6) * 0.05

# # Save
# output_path = os.path.join(leak_cleaned_path, "leak_cleaned.wav")
# sf.write(output_path, clean_data, sr)
# print(f"✅ 복소수 차감 방식 저장 완료: {output_path}")


#----------------------------------------------------------------------------------------------
cleaned = nr.reduce_noise(y=leak_data, y_noise=noise_data, sr=sr)
output_path = os.path.join(leak_cleaned_path, "cleaned_강아지.wav")
sf.write(output_path, cleaned, sr)
print(f"✅ 복소수 차감 방식 저장 완료: {output_path}")