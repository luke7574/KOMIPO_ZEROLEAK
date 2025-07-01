import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D plot)
import librosa
import librosa.display
import plotly.graph_objects as go
import scipy.io as sio


wav_path = "C:/Users/user/중부발전/M2_Leak/0613_0619/FH102/3024_20250613_033000/테스트/output_350_400.wav"
plot_path = "C:/Users/user/AI/KOMIPO_ZeroLeak/test/3D_FFT_PLOT/fft_3d_plot.mat"


data, samplerate = librosa.load(wav_path, sr=None, duration=5)
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



# ✅ 3D FFT 이미지 저장 함수
def save_fft_plot_3d(signal, samplerate, title, save_path, n_fft=1024, hop_length=512):
    # STFT 수행 (복소수 → 진폭)
    D = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(D)
    amplitude_db = librosa.amplitude_to_db(magnitude, ref=np.max)

    # 시간, 주파수 벡터 생성
    times = librosa.frames_to_time(np.arange(amplitude_db.shape[1]), sr=samplerate, hop_length=hop_length)
    freqs = librosa.fft_frequencies(sr=samplerate, n_fft=n_fft)
    T, F = np.meshgrid(times, freqs)

    # 3D 그래프 생성
    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(T, F, amplitude_db, cmap='viridis', linewidth=0, antialiased=False)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_zlabel("Amplitude (dB)")
    ax.set_title(title)

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✅ 3D FFT 그래프 저장 완료: {save_path}")


def save_fft_plot_3d_plotly(signal, samplerate, title, save_path_html, n_fft=1024, hop_length=512):
    # STFT → Magnitude → dB
    D = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(D)
    amplitude_db = librosa.amplitude_to_db(magnitude, ref=np.max)

    # 시간, 주파수 벡터 생성
    times = librosa.frames_to_time(np.arange(amplitude_db.shape[1]), sr=samplerate, hop_length=hop_length)
    freqs = librosa.fft_frequencies(sr=samplerate, n_fft=n_fft)

    # Plotly 3D Surface
    fig = go.Figure(data=[go.Surface(
        z=amplitude_db,  # 크기(dB)
        x=times,         # 시간축
        y=freqs,         # 주파수축
        colorscale='Viridis'
    )])

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='Time (s)',
            yaxis_title='Frequency (Hz)',
            zaxis_title='Amplitude (dB)'
        ),
        margin=dict(l=10, r=10, b=10, t=40)
    )

    # HTML 파일로 저장
    fig.write_html(save_path_html)
    print(f"✅ Plotly 3D FFT 그래프 저장 완료: {save_path_html}")

# save_fft_plot_3d_plotly(data, samplerate, "3D_FFT", plot_path)


def export_stft_to_mat(signal, samplerate, filename, n_fft=1024, hop_length=512):
    # STFT 계산
    D = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(D)
    amplitude_db = librosa.amplitude_to_db(magnitude, ref=np.max)

    # 시간/주파수 벡터
    times = librosa.frames_to_time(np.arange(amplitude_db.shape[1]), sr=samplerate, hop_length=hop_length)
    freqs = librosa.fft_frequencies(sr=samplerate, n_fft=n_fft)

    # .mat 파일로 저장
    sio.savemat(filename, {
        'amplitude_db': amplitude_db,
        'times': times,
        'freqs': freqs
    })
    print(f"✅ MATLAB용 .mat 파일 저장 완료: {filename}")

export_stft_to_mat(signal=data, samplerate=samplerate, filename=plot_path)