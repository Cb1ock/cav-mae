import os
import numpy as np
import argparse
import concurrent.futures

def process_audio(input_f):
    base_path = "/data/public_datasets/CelebV-Text/audio/celebvtext_audio"
    input_f = os.path.join(base_path, input_f)
    ext_len = len(input_f.split('/')[-1].split('.')[-1])
    audio_id = input_f.split('/')[-1][:-ext_len-1]
    output_f_1 = os.path.join(args.target_fold, audio_id + '_intermediate.wav')
    output_f_2 = os.path.join(args.target_fold, audio_id + '.wav')
    os.system('ffmpeg -i {:s} -vn -ar 16000 {:s}'.format(input_f, output_f_1)) # convert m4a to wav and resample
    os.system('sox {:s} {:s} remix 1'.format(output_f_1, output_f_2)) # extract the first channel
    os.remove(output_f_1) # remove the intermediate file

parser = argparse.ArgumentParser(description='Easy audio feature extractor')
parser.add_argument("-input_file_list", type=str, default='celebvtext_audio_list.csv', help="Should be a csv file of a single columns, each row is the input audio path.")
parser.add_argument("-target_fold", type=str, default='/data/public_datasets/CelebV-Text/audio/sample_audio/', help="The place to store the processed audio files.")
args = parser.parse_args()

input_filelist = np.loadtxt(args.input_file_list, delimiter=',', dtype=str)

if os.path.exists(args.target_fold) == False:
    os.makedirs(args.target_fold)

num_cores = os.cpu_count()
max_workers = num_cores//2
# Use a ThreadPoolExecutor to process the audio files in parallel
with concurrent.futures.ThreadPoolExecutor(max_workers) as executor:
    executor.map(process_audio, input_filelist)