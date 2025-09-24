import cv2
import csv
import mediapipe as mp
mp_holistic = mp.solutions.holistic

with open('random_sample.csv') as csvfile, open('mini_dataset.csv', 'w') as dataset_csv:
    reader = csv.reader(csvfile)
    writer = csv.writer(dataset_csv)
    next(reader)
    writer.writerow(['split', 'file', 'start_frame', 'end_frame', 'gloss'])

    for line in reader:
        filename = line[1]
        video = cv2.VideoCapture(f'../ASL_Citizen/videos/{filename}.mp4')
        fps = video.get(cv2.CAP_PROP_FPS)
        size = (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT)
        frame_num = 0
        begin_frame_num = 0
        end_frame_num = 0
        in_clip = False
        strikes = 0
        with mp_holistic.Holistic(min_detection_confidence=0.1, min_tracking_confidence=0.1) as holistic:
            while video.isOpened():
                if frame_num - begin_frame_num >= 3 * fps:
                    end_frame_num = frame_num
                    break

                ret, frame = video.read()
                if not ret:
                    break
                
                frame.flags.writeable = False
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(image)

                num_landmarks = [0, 0]
                if results.left_hand_landmarks:
                    for i in range(21):
                        lhand_pt = results.left_hand_landmarks.landmark[i]
                        if lhand_pt and lhand_pt.y <= 0.8:
                            num_landmarks[0] += 1

                if results.right_hand_landmarks:
                    for i in range(21):
                        rhand_pt = results.right_hand_landmarks.landmark[i]
                        if rhand_pt and rhand_pt.y <= 0.8:
                            num_landmarks[1] += 1
                
                if num_landmarks[0] > 10 or num_landmarks[1] > 10:
                    if not in_clip:
                        begin_frame_num = frame_num
                        in_clip = True
                    if strikes > 0:
                        strikes = 0
                elif in_clip:
                    strikes += 1
                    if strikes >= 12:
                        end_frame_num = frame_num
                        break

                frame_num += 1
        video.release()
        
        if end_frame_num == 0:
            end_frame_num = frame_num
        writer.writerow([line[0], filename, str(begin_frame_num), str(end_frame_num), line[2]])