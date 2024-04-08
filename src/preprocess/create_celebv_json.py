#  {
#    "video_id": "przrSPZgOkY",
#    "wav": "/data/sls/audioset/dave_version/audio/przrSPZgOkY.flac",
#    "video_path": "/data/sls/audioset/dave_version/image_mulframe/",
#    "labels": "/m/06mb1,/m/0jb2l,/m/0ngt1"
#   },

import json
import os

audio_dir = '/data/public_datasets/CelebV-Text/audio/sample_audio'
video_dir = '/data/public_datasets/CelebV-Text/video/frames'

audio_files = os.listdir(audio_dir)
video_dirs = os.listdir(video_dir)

print(f'Found {len(audio_files)} audio files and {len(video_dirs)} video directories')

data = []
for audio_file in audio_files:
    video_id = audio_file.split('.')[0]
    
    data.append({
        'video_id': video_id,
        'wav': os.path.join(audio_dir, audio_file),
        'video_path': video_dir,
        'labels': 'virtual label'
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