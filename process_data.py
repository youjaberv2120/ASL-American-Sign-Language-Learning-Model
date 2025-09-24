import cv2
import os
import csv
import mediapipe as mp
import pandas as pd

mp_holistic = mp.solutions.holistic

features_face = [70, 105, 107, 33, 160, 158, 133, 153, 144,
                 336, 334, 300, 362, 385, 387, 263, 373, 380,
                 78, 73, 11, 303, 308, 320, 315, 85, 90]

for filename in os.listdir('./ASL_Citizen/splits'):
    with open(f'./ASL_Citizen/splits/{filename}') as csv_file:
        reader = csv.reader(csv_file)
        next(reader)

        for row in reader:
            print(row)
            video_id = row[1].split('.')[0]
            word = row[2]

            video = cv2.VideoCapture(f'./ASL_Citizen/videos/{video_id}.mp4')
            fps = video.get(cv2.CAP_PROP_FPS)
            frame_num = 0
            begin_frame_num = 0
            in_clip = False
            strikes = 0

            data = pd.DataFrame()
            row = 0
            with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5, refine_face_landmarks=True) as holistic:
                while video.isOpened():
                    if frame_num - begin_frame_num > 3 * fps:
                        break

                    ret, frame = video.read()
                    if not ret:
                        break

                    frame.flags.writeable = False
                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = holistic.process(img)

                    if not results.left_hand_landmarks and not results.right_hand_landmarks and in_clip:
                        strikes += 1
                        if strikes == 2:
                            break
                    if (results.left_hand_landmarks or results.right_hand_landmarks) and not in_clip:
                        begin_frame_num = frame_num
                        in_clip = True

                    if in_clip:
                        id = 0
                        data.loc[row, 'frame'] = frame_num
                        if results.face_landmarks:
                            for i in features_face:
                                face_pt = results.face_landmarks.landmark[i]
                                if face_pt:
                                    data.loc[row, f'x{id}'] = face_pt.x
                                    data.loc[row, f'y{id}'] = face_pt.y
                                else:
                                    data.loc[row, f'x{id}'] = 0
                                    data.loc[row, f'y{id}'] = 0
                                id += 1
                        else:
                            for i in range(468):
                                data.loc[row, f'x{id}'] = 0
                                data.loc[row, f'y{id}'] = 0
                                id += 1

                        if results.left_hand_landmarks:
                            for i in range(21):
                                lhand_pt = results.left_hand_landmarks.landmark[i]
                                if lhand_pt:
                                    data.loc[row, f'x{id}'] = lhand_pt.x
                                    data.loc[row, f'y{id}'] = lhand_pt.y
                                else:
                                    data.loc[row, f'x{id}'] = 0
                                    data.loc[row, f'y{id}'] = 0
                                id += 1
                        else:
                            for i in range(21):
                                data.loc[row, f'x{id}'] = 0
                                data.loc[row, f'y{id}'] = 0
                                id += 1

                        if results.right_hand_landmarks:
                            for i in range(21):
                                rhand_pt = results.right_hand_landmarks.landmark[i]
                                if rhand_pt:
                                    data.loc[row, f'x{id}'] = rhand_pt.x
                                    data.loc[row, f'y{id}'] = rhand_pt.y
                                else:
                                    data.loc[row, f'x{id}'] = 0
                                    data.loc[row, f'y{id}'] = 0
                                id += 1
                        else:
                            for i in range(21):
                                data.loc[row, f'x{id}'] = 0
                                data.loc[row, f'y{id}'] = 0
                                id += 1
                        
                        row += 1
                    
                    frame_num += int(fps * 0.25)
                    video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                    if cv2.waitKey(1) == ord('q'):
                        break
            video.release()
            data.to_csv(f'./dataset/{video_id}.csv')