import subprocess
import os
import sys
# Pre...

video_path = '/Users/aryaman/research/FER_datasets/video/video_files'
all_files = os.listdir(video_path)
video_files = []
for fl in all_files:
    if fl[-4:] == ".mp4":
        video_files.append(fl)

for fl in video_files:
    input_path = video_path + '/' + fl
    output_path = video_path + '/' + fl.split('.')[0]
    subprocess.call(['scenedetect', '--input', input_path, 'detect-content',  'list-scenes', 'split-video',
                     '--output', output_path])
