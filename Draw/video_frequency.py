import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft, fftfreq
from pydub import AudioSegment
import pandas as pd

# 将MP3转换为WAV
def convert_mp3_to_wav(mp3_path, wav_path):
    audio = AudioSegment.from_mp3(mp3_path)
    audio.export(wav_path, format="wav")



def scale_audio(data, target_rms):
    current_rms = np.sqrt(np.mean(data ** 2))
    print(f"Current RMS: {current_rms}")
    scale_factor = target_rms / current_rms
    scaled_data = data * scale_factor
    return scaled_data


# 读取WAV文件并进行RMS缩放
def read_wav_file(female_wav_path, male_wav_path):
    female_sample_rate, female_data = wavfile.read(female_wav_path)
    male_sample_rate, male_data = wavfile.read(male_wav_path)

    # 确保数据为浮点数类型
    female_data = female_data.astype(np.float32)
    male_data = male_data.astype(np.float32)

    female_rms = np.sqrt(np.mean(female_data ** 2))
    male_rms = np.sqrt(np.mean(male_data ** 2))

    print(f"Original female RMS: {female_rms}, Original male RMS: {male_rms}")

    target_rms = male_rms
    female_data = scale_audio(female_data, target_rms)

    female_rms_new = np.sqrt(np.mean(female_data ** 2))
    print(f"Scaled female RMS: {female_rms_new}, Target RMS: {target_rms}")

    if female_sample_rate == male_sample_rate:
        return female_sample_rate, female_data, male_data
    else:
        raise ValueError("Sample rates do not match")


# 进行频谱分析
def perform_fft(sample_rate, data):
    # 只取单声道
    if len(data.shape) > 1:
        data = data[:, 0]

    N = len(data)
    yf = fft(data)
    xf = fftfreq(N, 1 / sample_rate)

    # 归一化幅值并转换为分贝
    min_db = 0
    yf = np.maximum(20 * np.log10(np.abs(yf[:N // 2]) / N), min_db)
    # yf = 20 * np.log10(np.abs(yf[:N // 2]) / N,mindb)

    return xf[:N // 2], yf


# 绘制频谱图
def plot_spectrum(female_xf, female_yf, male_xf, male_yf):
    plt.subplots(2, 1, sharex=True)

    # 平滑处理
    female_yf_smooth = smooth(female_yf)
    male_yf_smooth = smooth(male_yf)
    #两个图
    # plt.subplot(2, 1, 1)
    # plt.plot(female_xf, female_yf_smooth)
    # plt.grid()
    # # plt.title("Frequency Spectrum", loc='center')
    # plt.ylabel("Amplitude (dB)")
    # plt.ylim(0, 50)
    # plt.xlim(20, 500)
    #
    #
    # plt.subplot(2, 1, 2)
    # plt.plot(male_xf, male_yf_smooth, color='g')
    # # d = pd.DataFrame(dict({'Female_Frequency': female_xf, 'Female_Amplitude': female_yf_smooth}))
    # # d.to_csv('./Voice_Frequency_Female.csv', index=False)
    # # p = pd.DataFrame(dict({'Male_Frequency': male_xf, 'Male_Amplitude': male_yf_smooth}))
    # # p.to_csv('./Voice_Frequency_Male.csv', index=False)
    # print('male_xf', len(male_xf))
    # print('male_yf', len(male_yf))
    # plt.xlabel("Frequency (Hz)")
    # plt.ylabel("Amplitude (dB)")
    # plt.ylim(0, 50)
    # plt.xlim(20, 500)
    # plt.grid()

    # plt.savefig('end' + '.tif', dpi=600, bbox_inches='tight')
    #
    # plt.show()

    #一个图
    plt.figure(figsize=(6, 3))
    plt.gcf().subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.3)
    plt.plot(female_xf, female_yf_smooth,color='#F25CA1')
    plt.tick_params(labelsize=20)
    # plt.xlabel("Frequency (Hz)")
    # plt.ylabel("Amplitude (dB)")
    plt.ylim(0, 50)
    plt.xlim(20, 500)
    plt.grid()

    plt.savefig('Female' + '.png', dpi=600, bbox_inches='tight', pad_inches=0.05)
    plt.savefig('Female' + '.tif', dpi=600, bbox_inches='tight', pad_inches=0.05)
    plt.show()

    plt.figure(figsize=(6, 3))
    plt.gcf().subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.3)
    plt.plot(male_xf, male_yf_smooth,color='#4E94F1')
    plt.tick_params(labelsize=20)
    # plt.xlabel("Frequency (Hz)")
    # plt.ylabel("Amplitude (dB)")
    plt.ylim(0, 50)
    plt.xlim(20, 500)
    plt.grid()

    plt.savefig('Male' + '.png', dpi=600, bbox_inches='tight', pad_inches=0.05)
    plt.savefig('Male' + '.tif', dpi=600, bbox_inches='tight', pad_inches=0.05)
    plt.show()


def smooth(data, window_size=15):
    window = np.ones(int(window_size)) / float(window_size)

    return np.convolve(data, window, 'same')

def main(female_path, male_path):
    female_wav_path = "female_converted.wav"
    convert_mp3_to_wav(female_path, female_wav_path)
    male_wav_path = 'male_converted.wav'
    convert_mp3_to_wav(male_path, male_wav_path)

    sample_rate, female_data, male_data = read_wav_file(female_wav_path, male_wav_path)
    female_xf, female_yf = perform_fft(sample_rate, female_data)
    male_xf, male_yf = perform_fft(sample_rate, male_data)
    plot_spectrum(female_xf, female_yf, male_xf, male_yf)


# 调用
female_path = "../Data_Audio/end_female.mp3"
male_path = "../Data_Audio/end_male.mp3"
main(female_path, male_path)
