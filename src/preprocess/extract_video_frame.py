# -*- coding: utf-8 -*-
# @Time    : 3/13/23 2:27 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : extract_video_frame.py

import os.path
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torchvision.utils import save_image

from multiprocessing import Pool

preprocess = T.Compose([
    T.Resize(224),
    T.CenterCrop(224),
    T.ToTensor()])


def extract_frame(input_video_path, target_fold, extract_frame_num=10):
    # TODO: you can define your own way to extract video_id
    ext_len = len(input_video_path.split('/')[-1].split('.')[-1])
    video_id = input_video_path.split('/')[-1][:-ext_len-1]
    vidcap = cv2.VideoCapture(input_video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    # this is to avoid vggsound video's bug on not accurate frame count
    total_frame_num = min(int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)), int(fps * 10))
    for i in range(extract_frame_num):
        frame_idx = int(i * (total_frame_num/extract_frame_num))
        print('Extract frame {:d} from original frame {:d}, total video frame {:d} at frame rate {:d}.'.format(i, frame_idx, total_frame_num, int(fps)))
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)
        _, frame = vidcap.read()
        cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im)
        image_tensor = preprocess(pil_im)
        # save in 'target_path/frame_{i}/video_id.jpg'
        if os.path.exists(target_fold + f'/frame_{i}') == False:
            os.makedirs(target_fold + f'/frame_{i}')
        save_image(image_tensor, target_fold +f'/frame_{i}' +f'/{video_id}' + '.jpg')

def process_video(video_file, target_fold):
    print(f'Processing video {video_file}...')
    
    
    extract_frame(video_file, target_fold)
    
if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Python script to extract frames from a video, save as jpgs.")
    parser.add_argument("-input_file_list", type=str, default='/home/hao/Project/cav-mae/src/preprocess/celebvtext_video_list.csv', help="Should be a csv file of a single columns, each row is the input video path.")
    parser.add_argument("-target_fold", type=str, default='/data/public_datasets/CelebV-Text/video/frames', help="The place to store the video frames.")
    args = parser.parse_args()

    # note the first row (header) is skipped
    input_filelist = np.loadtxt(args.input_file_list, dtype=str, delimiter=',')

    file_path = '/data/public_datasets/CelebV-Text/video/celebvtext_video/'
    
    num_file = input_filelist.shape[0]
    print(f'Total {num_file} videos are input')

    num_cores = os.cpu_count()
    # create a multiprocessing Pool
    pool = Pool(processes=num_cores)
    video_paths = [(file_path + input_file, args.target_fold) for input_file in input_filelist]
    
    pool.starmap(process_video, video_paths)
