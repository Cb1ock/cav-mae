# -*- coding: utf-8 -*-
import os
import shutil
from tqdm import tqdm

base_path = '/data/public_datasets/CelebV-Text/video/frames_10frame_for_1video'
target_path = '/data/public_datasets/CelebV-Text/video/frames'

#在当前文件夹下新建十个文件夹
for i in range(10):
    dir_path = os.path.join(target_path, 'frames_{}'.format(i))
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print('Created directory: {}'.format(dir_path))

# 打开文件夹，并读取里面的每一个文件的名字
for folder in tqdm(os.listdir(base_path), desc='Processing folders', unit='folder'):
    folder_path = os.path.join(base_path, folder)
    file_name = folder
    #print('Processing folder: {}'.format(folder_path))
    
    for file in os.listdir(folder_path):
        num = file.split('_')[-1].split('.')[0]
        source_file_path = os.path.join(folder_path, file)
        target_file_path = os.path.join(target_path, 'frames_{}'.format(num), file_name + '.jpg')
        print(source_file_path)
        print(target_file_path)
        # mv bass_path/folder/frames_{0,...,9}.jpg to target_path/frames_{0,...,9}/file_name.jpg
        os.rename(source_file_path, target_file_path)
        #print('Copied file from {} to {}'.format(source_file_path, target_file_path))