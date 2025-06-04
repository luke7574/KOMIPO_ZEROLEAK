import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
import soundfile as sf

# 1. 오디오 로드
wav = 'C:/Users/user/Desktop/NELOW_AI_참고자료/중부발전/강아지 wav/161596_20231101_14_21_55_127_N.wav'
y, sr = librosa.load(wav, sr=16000)

# 2. STFT로 스펙트로그램 생성
S = np.abs(librosa.stft(y, n_fft=1024))  # (513,188)
S_db = librosa.amplitude_to_db(S, ref=np.max)  # (513,188)

# 3. NMF 적용 (3개의 구성 요소로 분해)
model = NMF(n_components=3, init='random', random_state=0, max_iter=1000)
W = model.fit_transform(S)
H = model.components_

# 4. 시각화: Component 3개 나란히 표시
fig, axs = plt.subplots(1, 3, figsize=(18, 4))

for i in range(3):
    component = np.outer(W[:, i], H[i, :])
    component_db = librosa.amplitude_to_db(component, ref=np.max)
    img = librosa.display.specshow(component_db, sr=sr, x_axis='time', y_axis='hz',
                                    cmap='magma', ax=axs[i])
    axs[i].set_title(f"Component {i+1}")
    axs[i].label_outer()  # 바깥쪽 Y축만 보이게

# colorbar는 첫 번째 이미지 기준으로 공통 적용
# cbar = fig.colorbar(img, ax=axs.ravel().tolist(), format='%+2.0f dB', location='right')

plt.suptitle("NMF Components (Dog Sound Analysis)", fontsize=14)
plt.tight_layout(rect=[0, 0, 0.97, 0.95])  # colorbar 공간 확보
plt.show()

# 5. 구성 요소별 재생성 및 저장
reconstructed_audio = np.zeros_like(y)  # 전체 합성용 배열

for i in range(3):
    # (1) 분해된 magnitude 복원
    component_mag = np.outer(W[:, i], H[i, :])

    # (2) 위상 정보 복원 (원래 S의 위상을 그대로 사용)
    component_phase = np.angle(S)
    component_complex = component_mag * np.exp(1j * component_phase)

    # (3) ISTFT로 시간 도메인 신호 복원
    y_component = librosa.istft(component_complex)

    # (4) WAV 파일로 저장
    out_path = f'C:/Users/user/Desktop/NELOW_AI_참고자료/중부발전/강아지 wav/component_{i+1}.wav'
    sf.write(out_path, y_component, sr)
    print(f"✅ 저장 완료: {out_path}")

    # 전체 합성용으로 누적
    reconstructed_audio[:len(y_component)] += y_component

# 6. 합성된 전체 오디오 저장
reconstructed_path = 'C:/Users/user/Desktop/NELOW_AI_참고자료/중부발전/강아지 wav/reconstructed_all_components.wav'

# 정규화 (클리핑 방지)
reconstructed_audio = reconstructed_audio / np.max(np.abs(reconstructed_audio) + 1e-6)
sf.write(reconstructed_path, reconstructed_audio, sr)
print(f"✅ 전체 합성 파일 저장 완료: {reconstructed_path}")


