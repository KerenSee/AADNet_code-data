import numpy as np
import librosa
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt
import soundfile as sf
from matplotlib import rcParams
from pydub import AudioSegment

config = {
    "font.family": 'serif',  # 衬线字体
    "font.size": 10,  # 相当于小四大小
    "font.serif": ['SimSun'],  # 宋体
    "mathtext.fontset": 'stix',  # matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
    'axes.unicode_minus': False  # 处理负号，即-号
}
# 配置当前终端
rcParams.update(config)

audio_data = 'female_converted.wav'
x, fs = librosa.load(audio_data, sr=None)
N = x.shape[0]
xx = np.linspace(0, 1, N)  # 将0到1平分成N份

# FFT
fft_y = fft(x)

# 求相角
phase_x = np.angle(fft_y)

# print(len(phase_x))
# 取复数的绝对值，即复数的模
mag_x = np.abs(fft_y)

# 使用模和相角构建复数序列
new_spec = mag_x * np.exp(1j * phase_x)
# IFFT
res = ifft(new_spec)
out = np.real(res)  # 取实数部分

# 保存成新的音频文件
sf.write("out_num1.wav", out, fs)

# 显示波形图
plt.plot(xx, out)
plt.ylim(-0.5,0.5)
plt.title('波形图')
plt.xlabel('时间')
plt.ylabel('响度')
plt.show()