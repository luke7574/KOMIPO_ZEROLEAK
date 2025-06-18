import torchaudio
import torch
from asteroid.models import ConvTasNet
import os
import soundfile as sf
import numpy as np


# ✅ 1. 사전학습된 ConvTasNet 모델 불러오기
model = ConvTasNet.from_pretrained("JorisCos/ConvTasNet_Libri2Mix_sepclean_16k")

# ✅ 2. 오디오 파일 로드
input_wav_path = "C:/Users/user/AI/KOMIPO_ZeroLeak/test/convtasnet_test/190228_20250611_10_27_09_126_N.wav"
waveform, sr = torchaudio.load(input_wav_path)

wav_filename = os.path.splitext(os.path.basename(input_wav_path))[0]

# ✅ 3. 입력 전처리: 모노로 변환
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)

# ✅ 4. 모델에 입력
with torch.no_grad():
    separated_sources = model.separate(waveform)

# ✅ 5. 출력 디렉토리 및 개별 저장 폴더 생성
base_output_dir = "C:/Users/user/AI/KOMIPO_ZeroLeak/test/convtasnet_test_clean"
target_folder = os.path.join(base_output_dir, wav_filename)
os.makedirs(target_folder, exist_ok=True)


# ✅ 각 분리 음성 저장 및 3초 단위 분할 저장
split_duration_sec = 3
split_samples = sr * split_duration_sec

# ✅ 6. 원본 waveform 길이까지만 잘라서 저장
original_len = waveform.shape[1]  # 원본 오디오 샘플 수
sources = separated_sources[0, :, :original_len].cpu().numpy()  # shape: (2, samples)
for i, data  in enumerate(sources):
    if data.ndim != 1:
        data = data.reshape(-1)
    data = data.astype(np.float32)

    total_samples = len(data)
    num_parts = int(np.ceil(total_samples / split_samples))

    for i, data in enumerate(sources):
        data = data.astype(np.float32)
        out_path = os.path.join(target_folder, f"{wav_filename}_{i+1}.wav")
        try:
            sf.write(out_path, data, sr, format='WAV', subtype='PCM_16')
            print(f"✅ 저장 완료: {out_path}")
        except Exception as e:
            print(f"❌ 저장 실패: {out_path} → {e}")