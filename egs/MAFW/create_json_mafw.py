import json
import os

audio_dir = '/data/chenghao/MAFW/data/sample_audio'
video_dir = '/data/chenghao/MAFW/data/frames'
anno_dir = '/data/chenghao/MAFW/anno/single_anno.txt'  # replace with your actual label file path
audio_files = os.listdir(audio_dir)
video_dirs = os.listdir(video_dir)

print(f'Found {len(audio_files)} audio files and {len(video_dirs)} video directories')

# Read labels from txt file
with open(anno_dir, 'r') as f:
    lines = f.readlines()
labels = {line.split()[0].split('.')[0]: line.split()[1] for line in lines}

import csv

# Create a dictionary to store unique labels and their indices
label_dict = {label: i for i, label in enumerate(set(labels.values()))}
print(label_dict)

# Create a CSV file
with open('class_labels_indices_mafw.csv', 'w', newline='') as csvfile:
    fieldnames = ['index', 'mid', 'display_name']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for label, index in label_dict.items():
        writer.writerow({'index': index, 'mid': index, 'display_name': label})

# Replace each label in the labels list with its corresponding index
labels = {key: label_dict[value] for key, value in labels.items()}
print(labels)

data = []
for audio_file in audio_files:
    video_id = audio_file.split('.')[0]
    
    if video_id in labels:
        data.append({
            'video_id': video_id,
            'wav': os.path.join(audio_dir, audio_file),
            'video_path': video_dir,
            'labels': labels[video_id]
        })


import random

# Shuffle data
random.shuffle(data)

# Calculate the size of the training set
train_size = int(len(data) * 0.9)
print('we have', train_size, 'training samples')

# Split the data
train_data = data[:train_size]
test_data = data[train_size:]

# Save training data as JSON
with open('train_data.json', 'w') as f:
    json.dump({'data': train_data}, f, indent=1)

# Save testing data as JSON
with open('test_data.json', 'w') as f:
    json.dump({'data': test_data}, f, indent=1)

import csv


