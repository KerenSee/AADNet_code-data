import os
import random
from datetime import datetime

# 设置文件夹路径
male_folder = 'Audio/male/'
female_folder = 'Audio/female/'
data_folder = 'data'

# 随机生成男女标签
is_male = random.choice([True, False])

# 随机选择左右声道的文件名
left_file = random.choice(os.listdir(male_folder if is_male else female_folder))
right_file = random.choice(os.listdir(female_folder if is_male else male_folder))

# 生成当前时间字符串
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# 生成记录文本文件的路径
log_file_path = os.path.join(data_folder, f'{current_time}_log.txt')




# 写入记录信息到文本文件
with open(log_file_path, 'w') as log_file:
    log_file.write(f'Timestamp: {current_time}\n')
    log_file.write(f'Left Channel: {"Male" if is_male else "Female"} - {left_file}\n')
    log_file.write(f'Right Channel: {"Female" if is_male else "Male"} - {right_file}\n')

print(f'Recording information saved to: {log_file_path}')
