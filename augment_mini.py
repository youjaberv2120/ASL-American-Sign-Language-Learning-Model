import csv
import ffmpeg
import random
import numpy as np

def augment_file(id):
    x_shifts = random.sample(range(-250, 250), 10)
    y_shifts = random.sample(range(-150, 150), 10)
    scales = np.random.uniform(0.5, 1, 10)
    blurs = np.random.permutation(10)

    for i in range(10):
        {
            ffmpeg.input(f'../ASL_Citizen/videos/{id}.mp4')
            .filter('pad', f'iw/{scales[i]}', f'ih/{scales[i]}', f'(ow-iw)/2', f'(oh-ih)/2')
            .filter('scale', f'iw*{scales[i]}', f'ih*{scales[i]}')
            .filter('crop', f'iw-{abs(x_shifts[i])}', f'ih-{abs(y_shifts[i])}', abs(min(0, x_shifts[i])), abs(min(0, y_shifts[i])))
            .filter('pad', f'iw+{abs(x_shifts[i])}', f'ih+{abs(y_shifts[i])}', max(0, x_shifts[i]), max(0, y_shifts[i]))
            .filter('gblur', blurs[i])
            .output(f'../augmented_dataset/{id}_{i + 1}.mp4').run(overwrite_output=True)
        }

with open('mini_dataset.csv') as in_csv, open('augmented_mini_dataset.csv', 'w') as out_csv:
    reader = csv.reader(in_csv)
    writer = csv.writer(out_csv)
    next(reader)
    writer.writerow(['split', 'file', 'gloss'])
    counter = 0

    for line in reader:
        split = line[0]
        id = line[1]
        ffmpeg.input(f'../ASL_Citizen/videos/{id}.mp4').output(f'../augmented_dataset/{id}.mp4').run(overwrite_output=True)
        writer.writerow(line)
        if split == 'train' or split == 'val':
            augment_file(id)
            for i in range(10):
                writer.writerow([line[0], f'{id}_{i + 1}', line[2], line[3], line[4]])
        counter += 1
        if counter % 100 == 0:
            print(counter)