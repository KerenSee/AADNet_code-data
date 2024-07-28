import mne
import numpy as np
import os
from scipy.signal import welch
from pydub import AudioSegment
import librosa
import librosa.display
import matplotlib.pyplot as plt


def eeg_process():
    data_folder = "Newdataset/"
    file_names = ["Female1-post.set", "Female2-post.set", "Female3-post.set", "Data1-post.set", "Data2-post.set",
                  "Male1-post.set", "Female7-post.set", "Female8-post.set", "Male2-post.set", "Male4-post.set",
                  "Male5-post.set", "Male6-post.set"]

    eeg_data_list = []
    for file_name in file_names:
        file_path = os.path.join(data_folder, file_name)
        raw = mne.io.read_raw_eeglab(file_path, preload=True)

        eeg_data = raw.get_data()[:32, :]
        eeg_data_list.append(eeg_data)
    all_eeg_data = np.stack(eeg_data_list, axis=0)

    fs = 500
    time = 5
    n = all_eeg_data.shape[-1] // (fs * time)
    data = np.zeros((all_eeg_data.shape[0], n, all_eeg_data.shape[1], fs * time))
    for i in range(n):
        data[:, i, :, :] = all_eeg_data[:, :, i * fs * time:(i + 1) * fs * time]
    # print(data.shape)

    person_index = 5
    n = int((34000 / 500) // 5)
    eeg = data[person_index, :, :, :]
    fs = 500
    frequencies, power_spectrum = welch(eeg, fs, nperseg=1024)  # 计算功率谱密度

    # plt.figure(figsize=(12, 6))
    # plt.plot(frequencies, 10 * np.log10(power_spectrum[0]))
    # plt.title('Power Spectrum of EEG Signal')
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Power/Frequency (dB/Hz)')
    # plt.grid(True)
    # plt.show()

    return power_spectrum, n


# power_spectrum（eeg）， db_spec[:, start_frame:end_frame]（wav）


def wav_process(file, n):
    mp3_file = AudioSegment.from_mp3(file)
    num_channels = mp3_file.channels
    print(f"Number of channels: {num_channels}")

    mp3_file.export('output.wav', format='wav')

    wav_file = AudioSegment.from_file('output.wav', format='wav')
    wav_file_path = 'output.wav'
    audio, sr = librosa.load(wav_file_path)
    stft = librosa.stft(audio)
    spectrogram = np.abs(stft) ** 2
    db_spec = librosa.power_to_db(spectrogram, ref=np.max)

    start_time = 0
    end_time = 5
    start_frame = int(librosa.time_to_frames(start_time, sr=sr))
    end_frame = int(librosa.time_to_frames(end_time, sr=sr))

    wav = np.zeros((n, db_spec[:, start_frame:end_frame].shape[0], db_spec[:, start_frame:end_frame].shape[1]))
    for i in range(n):
        wav[i, :, :] = db_spec[:, end_frame * i:end_frame * (i + 1)]  # 0-5,
    # print(data.shape)

    plt.figure(figsize=(10, 6))
    librosa.display.specshow(db_spec[:, start_frame:end_frame], sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%2.0f dB')
    plt.title('Spectrogram of Audio Signal (0-5 seconds)')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    return wav


def corr(eeg_pca, wav_pca):
    cor = np.zeros(eeg_pca.shape[0])
    for i in range(eeg_pca.shape[0]):
        cor[i] = np.corrcoef(eeg_pca[i], wav_pca[i])[0, 1]
    return cor


eeg, n = eeg_process()
wav = wav_process('./Audio/1-Male.mp3', n)
wav_noise = wav_process('./Audio/Noise-1-Female.mp3', n)
print(n, wav.shape, wav_noise.shape, eeg.shape)

from sklearn.decomposition import PCA

n_components = 10
pca = PCA(n_components=n_components)
eeg_pca = pca.fit_transform(eeg.reshape(n, -1))
wav_pca = pca.fit_transform(wav.reshape(n, -1))
wave_noise_pca = pca.fit_transform(wav_noise.reshape(n, -1))
c = corr(eeg_pca, wav_pca)
c_noise = corr(eeg_pca, wave_noise_pca)
print('main attention', c)
print('noise         ', c_noise)
c_sum = 0
# for i in c:
#     c_sum += i
# c_noise_sum = 0
# for i in c_noise:
#     c_noise_sum += i
print('main attention max is:', max(abs(c)))
print('noise max is         :', max(abs(c_noise)))

# plt.show()
import numpy as np

def shannon_entropy(arr):
    # 香农熵的计算
    probabilities = arr / np.sum(arr)
    non_zero_probabilities = probabilities[probabilities != 0]
    entropy = -np.sum(non_zero_probabilities * np.log2(non_zero_probabilities))
    return entropy

def renyi_entropy(arr, alpha):
    # Rényi熵的计算
    probabilities = arr / np.sum(arr)
    non_zero_probabilities = probabilities[probabilities != 0]
    entropy = 1 / (1 - alpha) * np.log2(np.sum(non_zero_probabilities**alpha))
    return entropy
#########################################################################
#计算熵值
#读取array
array1 = np.abs(c)
array2 = np.abs(c_noise)
# 计算香农熵
entropy1 = shannon_entropy(array1)
entropy2 = shannon_entropy(array2)

print("Shannon Entropy (Target):", entropy1)
print("Shannon Entropy (Noise):", entropy2)

# 计算Rényi熵，取alpha为2，可调整
alpha_value = 2
renyi_entropy1 = renyi_entropy(array1, alpha_value)
renyi_entropy2 = renyi_entropy(array2, alpha_value)

print(f"Rényi Entropy (Target, alpha={alpha_value}):", renyi_entropy1)
print(f"Rényi Entropy (Noise, alpha={alpha_value}):", renyi_entropy2)
