import random
import os
import pyaudio
import numpy as np
import json
import time
from pydub.playback import play
from pydub import AudioSegment
def preAudio(audio, length, db):
    target_duration = length * 1000
    audio = audio[:target_duration]
    audio = np.array(audio.get_array_of_samples())
    audio = audio / audio.std(ddof=1)
    audio = (audio * db).astype(np.int16)
    return audio
start_male = AudioSegment.from_file('D:/P/TFProject/Data_Audio/male.mp3', format='mp3')
start_female = AudioSegment.from_file('D:/P/TFProject/Data_Audio/female.mp3', format='mp3')
end_male = AudioSegment.from_file('D:/P/TFProject/Data_Audio/end_male.mp3', format='mp3')
end_female = AudioSegment.from_file('D:/P/TFProject/Data_Audio/end_female.mp3', format='mp3')
male_audio = ['D:/P/TFProject/Data_Audio/Audio/male/' + i for i in os.listdir('D:/P/TFProject/Data_Audio/Audio/male')]
female_audio = ['D:/P/TFProject/Data_Audio/Audio/female/' + i for i in os.listdir('D:/P/TFProject/Data_Audio/Audio/female')]
random.shuffle(male_audio)
random.shuffle(female_audio)
audio_length = 70  # s
db = 1000
log = []
name = input('please enter the name')
for i in range(5):
    male_i = AudioSegment.from_file(male_audio[i], format='mp3')
    female_i = AudioSegment.from_file(female_audio[i], format='mp3')

    male = preAudio(male_i, audio_length, db)
    female = preAudio(female_i, audio_length, db)

    att = random.sample(['male', 'female'], 1)[0]
    att_location = random.sample(['left', 'right'], 1)[0]
    log_i = {
        'i': i,
        'length': audio_length,
        'male_i': male_audio[i],
        'female_i': female_audio[i],
        'att': att,
        'att_location': att_location,
    }
    print(log_i)
    if 'male' == att:
        play(start_male)
    else:
        play(start_female)
    time.sleep(1)


    stream = pyaudio.PyAudio().open(format=pyaudio.paInt16, channels=2, rate=48000, output=True)

    audio_data = np.array([])
    if ('male' == att and 'left' == att_location) or ('female' == att and 'right' == att_location):
        audio_data = np.column_stack((male, female)).flatten()
    elif ('male' == att and 'right' == att_location) or ('female' == att and 'left' == att_location):
        audio_data = np.column_stack((female, male)).flatten()

    stream.stop_stream()
    stream.close()
    time.sleep(1)

    if 'male' == att:
        play(end_male)
    else:
        play(end_female)

    log.append(log_i)
    input('press any key to continue')
    time.sleep(1)

log = json.dumps(log)
with open('./log/{}-{}.json'.format(name, int(time.time())), 'w', encoding='utf-8') as file:
    file.write(log)
print('end')