import cv2
import mediapipe as mp
import numpy as np
import keras

new_model = keras.models.load_model('C:/Users/SMAI-JH/Documents/Coding/capstone03/ML3/two_DNN_duck_flower2.h5')

max_num_hands = 2
gesture = {
    0:'butterfly', 1:'cat', 2:'snail', 3:'deer', 4:'heart', 5:'ori',
    6:'sun', 7:'house', 8:'tree', 9:'rock', 10:'flower', 11:'dog1', 12:'dog2', 13:'dog3'
}

'''
gesture = {
    0=0:'Butterfly', 2=1:'snail', 3=2:'deer', 4=3:'heart', 5=4:'elephant'
    6=5:'sun', 7=6:'house', 8=7:'tree', 10=8:'flower'
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

    tmp_list = []
    #mp_drawing.draw_landmarks(img, result.multi_hand_landmarks, mp_hands.HAND_CONNECTIONS)

    if result.multi_hand_landmarks is not None:

        for num, hand in enumerate(result.multi_hand_landmarks):
            tmp_list.append(hand.landmark)
            mp_drawing.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)
        
        if (len(tmp_list)==2):
            l_count = 0
            r_count = 0
            left_joint_np = np.zeros((21, 3))
            right_joint_np = np.zeros((21, 3))

            for data_point in tmp_list[0]:
                left_joint_np[l_count] = [data_point.x, data_point.y, data_point.z]
                l_count += 1
            
            for data_point in tmp_list[1]:
                right_joint_np[r_count] = [data_point.x, data_point.y, data_point.z]
                r_count += 1

            # Compute angles between joints ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
            l_v1 = left_joint_np[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
            l_v2 = left_joint_np[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
            l_v = l_v2 - l_v1 # [20,3]
            # Normalize v
            l_v = l_v / np.linalg.norm(l_v, axis=1)[:, np.newaxis]

            # Get angle using arcos of dot product
            l_angle = np.arccos(np.einsum('nt,nt->n',
                l_v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                l_v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

            l_angle = np.degrees(l_angle) # Convert radian to degree ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ

            # Compute angles between joints ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
            r_v1 = right_joint_np[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
            r_v2 = right_joint_np[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
            r_v = r_v2 - r_v1 # [20,3]
            # Normalize v
            r_v = r_v / np.linalg.norm(r_v, axis=1)[:, np.newaxis]

            # Get angle using arcos of dot product
            r_angle = np.arccos(np.einsum('nt,nt->n',
                r_v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                r_v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

            r_angle = np.degrees(r_angle) # Convert radian to degree ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ

            l_data = np.array([l_angle], dtype=np.float32)
            r_data = np.array([r_angle], dtype=np.float32)
            data = np.concatenate((l_data, r_data), axis=1)
            print(len(data))


            y_prob = new_model.predict(data, verbose=0)
            
            predicted = 100

            for i in y_prob[0]:
                if i >= 0.8:
                    predicted = y_prob.argmax(axis=-1)
                
            print(predicted)

            if predicted==0:
                cv2.putText(img, "Butterfly", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)

            if predicted==1:
                cv2.putText(img, "snail", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)

            if predicted==2:
                cv2.putText(img, "deer", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)

            if predicted==3:
                cv2.putText(img, "heart", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)

            if predicted==4:
                cv2.putText(img, "elephant", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)

            if predicted==5:
                cv2.putText(img, "sun", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)

            if predicted==6:
                cv2.putText(img, "house", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)

            if predicted==7:
                cv2.putText(img, "tree", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)

            if predicted==8:
                cv2.putText(img, "flower", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)

            if predicted==100:
                cv2.putText(img, "Normal", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('Filter', img)
    if cv2.waitKey(1) == ord('q'):
        break