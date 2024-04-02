import os
import csv

# 文件夹路径
folder_path = '/data/public_datasets/CelebV-Text/video/celebvtext_6'

# 获取文件夹中的所有文件名
file_names = os.listdir(folder_path)

# csv文件路径
csv_file_path = '/home/hao/Project/cav-mae/src/preprocess/celebvtext_video_list.csv'

# 打开csv文件并创建一个csv写入器
with open(csv_file_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    # 将文件名写入csv文件
    for file_name in file_names:
        writer.writerow([file_name])