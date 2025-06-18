import torchaudio
from demucs.apply import apply_model
from demucs.pretrained import get_model
import os

# 모델 불러오기
model = get_model(name="htdemucs")  # 또는 htdemucs_ft / demucs48_hq 등

# 파일 로드 및 리샘플링
input_wav_path = "C:/Users/user/AI/KOMIPO_ZeroLeak/test/demucs_test/전기/153606_20231010_09_11_01_127_N.wav"
waveform, sr = torchaudio.load(input_wav_path)

# (채널, 시간) → (1, 채널, 시간): 배치 차원 추가
waveform = waveform.unsqueeze(0)
# 리샘플링
if sr != 44100:
    resampler = torchaudio.transforms.Resample(sr, 44100)
    waveform = resampler(waveform.squeeze(0)).unsqueeze(0)

# ✅ 1채널 → 2채널 (Demucs는 스테레오 기대)
if waveform.shape[1] == 1:
    waveform = waveform.repeat(1, 2, 1)    

# 분리 실행
sources = apply_model(model, waveform, device='cpu')  # shape: [num_sources, channels, time]

# 저장 (batch 차원 제거 후 저장)
output_dir = "C:/Users/user/AI/KOMIPO_ZeroLeak/test/demucs_test/전기"

source_names = ["drums", "bass", "other", "vocals"]
for i, name in enumerate(source_names):
    audio = sources[0, i]  # shape: (2, samples)
    out_path = os.path.join(output_dir, f"{name}.wav")
    torchaudio.save(out_path, audio, 44100)
    print(f"✅ 저장 완료: {out_path}")
