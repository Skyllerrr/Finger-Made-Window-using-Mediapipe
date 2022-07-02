import cv2
import mediapipe as mp
import numpy as np
import keras

new_model = keras.models.load_model('C:/Users/SMAI-JH/Documents/Coding/capstone03/ML3/one_DNN_stop.h5')

max_num_hands = 1
gesture = {
    0:'butterfly', 1:'cat', 2:'snail', 3:'deer', 4:'heart', 5:'elephant',
    6:'sun', 7:'house', 8:'tree', 9:'rock', 10:'flower', 11:'dog1', 12:'dog2', 13:'dog3'
}
'''
gesture = {
    1=0:'cat', 9=1:'rock', 11=2:'dog1', 12=3:'dog2', 13=4:'dog3'
}
'''

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        continue

    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = hands.process(img)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks is not None:
        

        for res in result.multi_hand_landmarks:

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            joint = np.zeros((21, 3))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z]

            # Compute angles between joints
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
            v = v2 - v1 # [20,3]
            # Normalize v
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # Get angle using arcos of dot product
            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

            angle = np.degrees(angle) # Convert radian to degree

            # Inference gesture
            data = np.array([angle], dtype=np.float32)

            y_prob = new_model.predict(data, verbose=0)
            
            predicted = 100
            # 분류한 정확도가 일정 정확도 이상이면 라벨로 분류
            for i in y_prob[0]:
                if i >= 0.9:
                    predicted = y_prob.argmax(axis=-1)

            if predicted==0:
                cv2.putText(img, "cat", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)

            if predicted==1:
                cv2.putText(img, "rock", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)

            if predicted==2:
                cv2.putText(img, "dog1", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)

            if predicted==3:
                cv2.putText(img, "dog2", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)

            if predicted==4:
                cv2.putText(img, "dog3", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)

            if predicted==5:
                cv2.putText(img, "STOP!!!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)

            if predicted==100:
                cv2.putText(img, "Normal", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('Filter', img)
    if cv2.waitKey(1) == ord('q'):
        break