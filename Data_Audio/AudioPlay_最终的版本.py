import time
s0 = time.time()
import sys
from pydub import AudioSegment
import pyaudio
import numpy as np
import random
from pynput import keyboard
import os
from datetime import datetime
from pydub.playback import play
import time


male_files = np.array(['D:/P/TFProject/Data_Audio/Audio/male/'+i for i in os.listdir('D:/P/TFProject/Data_Audio/Audio/male/')])
female_files = np.array(['D:/P/TFProject/Data_Audio/Audio/female/'+i for i in os.listdir('D:/P/TFProject/Data_Audio/Audio/female/')])
log_file_path = r'D:\WorkSpace\实验记录\fNIRS\被试——杨劲峰\record\record.txt'
log_directory = r'D:\WorkSpace\实验记录\fNIRS\被试——杨劲峰\record'
target_duration = 70 * 1000
rest_time = 15


left_right_array = np.array([0]*8 + [1]*8)
gender_array = np.array([0]*8 + [1]*8)
random.shuffle(left_right_array)
random.shuffle(gender_array)
# 左右矩阵（左 == 1）
# 男女矩阵（男 == 1）
print(left_right_array)
print(gender_array)
for trial_index, male_file in enumerate(male_files, start=1):
    log_trial_path = os.path.join(log_directory, f'trial_{trial_index}.txt')
    is_male = gender_array[trial_index - 1]
    is_left = left_right_array[trial_index - 1]

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    index_male = np.random.randint(len(male_files))
    index_female = np.random.randint(len(female_files))

    # 通过索引获取随机选择的音频文件路径
    selected_male_file = male_files[index_male]
    selected_female_file = female_files[index_female]

    #检查是否已经在日志文件中记录
    with open(log_file_path, 'r') as log_file:
        log_contents = log_file.read()
        while selected_male_file in log_contents or selected_female_file in log_contents:
            index_male = np.random.randint(len(male_files))
            index_female = np.random.randint(len(female_files))
            selected_male_file = male_files[index_male]
            selected_female_file = female_files[index_female]

    # 将选择的文件路径添加到日志文件，采用追加写入的方式
    with open(log_file_path, 'a') as log_file:
        log_file.write(f'{current_time}: Male File: {selected_male_file}, Female File: {selected_female_file}\n')
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=2,  # 双声道，左耳和右耳
                    rate=48000,  # 采样率
                    output=True)
    male_audio = AudioSegment.from_file('male.mp3', format="mp3")
    female_audio = AudioSegment.from_file('female.mp3', format="mp3")
    # if is_male:
    #     play(male_audio)
    # else:
    #     play(female_audio)

    if 1 == is_left and 1 == is_male:
        left_audio = AudioSegment.from_file(male_files[index_male], format="mp3")
        right_audio = AudioSegment.from_file(female_files[index_female], format="mp3")
    elif 1 == is_left and 0 == is_male:
        left_audio = AudioSegment.from_file(female_files[index_male], format="mp3")
        right_audio = AudioSegment.from_file(male_files[index_female], format="mp3")
    elif 0 == is_left and 0 == is_male:
        left_audio = AudioSegment.from_file(male_files[index_male], format="mp3")
        right_audio = AudioSegment.from_file(female_files[index_female], format="mp3")
    else:
        left_audio = AudioSegment.from_file(female_files[index_male], format="mp3")
        right_audio = AudioSegment.from_file(male_files[index_female], format="mp3")

    with open(log_trial_path, 'w') as log_file:
        log_file.write(f'Timestamp: {current_time}\n')
        log_file.write(f'Left Channel: {"Male" if is_male else " Female"} - {male_files[index_male]}\n')
        log_file.write(f'Right Channel: {"Female" if is_male else " Male"} - {female_files[index_female]}\n')
        if 1 == is_left and 1 == is_male:
            log_file.write('The target is male\nThe target is left\n')
        elif 1 == is_left and 0 == is_male:
            log_file.write('The target is female\nThe target is left\n')
        elif 0 == is_left and 0 == is_male:
            log_file.write('The target is female\nThe target is right\n')
        else:
            log_file.write('The target is male\nThe target is right\n')

    print(f'Recording information saved to: {log_file_path}')

    left_audio = left_audio[:target_duration]
    right_audio = right_audio[:target_duration]

    # 获取左耳和右耳音频数据
    left_audio_data = np.array(left_audio.get_array_of_samples())
    right_audio_data = np.array(right_audio.get_array_of_samples())

    # 确保左右声道的音频数据具有相同的长度
    max_length = max(len(left_audio_data), len(right_audio_data))
    left_audio_data = np.pad(left_audio_data, (0, max_length - len(left_audio_data)), mode='constant')
    right_audio_data = np.pad(right_audio_data, (0, max_length - len(right_audio_data)), mode='constant')

    db = 1000
    left_audio_data = left_audio_data / left_audio_data.std(ddof=1)
    left_audio_data = (left_audio_data * db).astype(np.int16)
    right_audio_data = right_audio_data / right_audio_data.std(ddof=1)
    right_audio_data = (right_audio_data * db).astype(np.int16)


    # 监听F7
    # print('Enter F7')
    #
    # def on_key_release(key):
    #     if key == keyboard.Key.f7:
    #         print('Started...')
    #         return False
    #
    # listener = keyboard.Listener(on_press= on_key_release, on_release=on_key_release)
    # listener.start()
    # listener.join()

    # 生成随机数，决定左右声道
    # random_channel = random.choice(['left', 'right'])
    # 合并左耳和右耳音频数据
    # if random_channel == 'left':
    combined_audio_data = np.column_stack((left_audio_data, right_audio_data)).flatten()
    # else:
    # combined_audio_data = np.column_stack((right_audio_data, left_audio_data)).flatten()
    # 播放音频
    r = 32
    s1 = time.time()
    if trial_index==1:
        time.sleep(r-(s1-s0))
        t = time.time()-s0

    else:
        time.sleep(r-(s1-s2))
        t = time.time()-s2

    sss1 = time.time()
    if is_male:
        play(male_audio)
    else:
        play(female_audio)

    time.sleep(8-(time.time()-sss1))


    print(trial_index,'-trail-before-time',t)
    s = time.time()
    stream.write(combined_audio_data.tobytes())
    # 生成一个文本，他记录当前时间、播放的左边声道文件是哪个，右边是哪个。
    # 等待音频播放完成

    stream.stop_stream()
    stream.close()
    # 关闭PyAudio
    p.terminate()
    s2 = time.time()
    time.sleep(71-(s2-s))
    print(trial_index,'-trail-play-time',time.time()-s)


    print(f'Recording information saved to: {log_trial_path}')
    time.sleep(1)
    end_male_audio = AudioSegment.from_file('end_male.mp3', format="mp3")
    end_female_audio = AudioSegment.from_file('end_female.mp3', format="mp3")
    if is_male:
        play(end_male_audio)
    else:
        play(end_female_audio)

    time.sleep(rest_time)

    if trial_index==12:
        s3 = time.time()
        print('last-trials-after-time',s3-s2)
print(s3-s0)
# with open(log_file_path.split('record.txt')[0]+'output.txt','w') as f:
#     sys.stdout = f
# sys.stdout = sys.__stdout__
