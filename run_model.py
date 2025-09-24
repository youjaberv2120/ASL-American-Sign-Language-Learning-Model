# remember to set "export PYTORCH_ENABLE_MPS_FALLBACK=1" before running

import numpy as np
import cv2
import torch
from torchvision.transforms import v2
from concurrent.futures import ThreadPoolExecutor
from model import Model

# model
model = Model()
saved = torch.load('./models/v4/best.pth', map_location=torch.device('mps'))
model.load_state_dict(saved['model'])
model = model.to('mps')
model = model.eval()

# label to class
classes = []
with open('sample_classes.txt') as labels_file:
    content = labels_file.readlines()
    for line in content:
        classes.append(line[:-1])

def predict(frames):
    # process data
    process = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
        v2.Resize((224, 224))
    ])
    frame_list = np.linspace(0, len(frames) - 1, 16).round().astype(int)
    processed_frames = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        processed_frames = executor.map(lambda frame_num: process(frames[frame_num]), frame_list)

    data = torch.empty(1, 16, 3, 224, 224)
    for i, f in enumerate(processed_frames):
        data[0][i] = f
    data = data.permute(0, 2, 1, 3, 4)

    # predict
    data = data.to('mps')
    outputs = model(data)
    post_act = torch.nn.Softmax(dim=1)
    pred = post_act(outputs).topk(k=3).indices[0]
    print(classes[pred[0]], classes[pred[1]], classes[pred[2]])

# video input
video = cv2.VideoCapture(0)
fps = video.get(cv2.CAP_PROP_FPS)
i = 0
frames = []
while video.isOpened():
    if i == 3 * fps:
        predict(frames)
        break

    ret, frame = video.read()
    if not ret:
        break
    frames.append(frame)
    cv2.imshow('video', frame)
    if cv2.waitKey(1) == ord('q'):
        break
    i += 1