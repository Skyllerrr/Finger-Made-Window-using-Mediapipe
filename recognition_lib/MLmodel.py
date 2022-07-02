from turtle import position
import cv2
#import joblib
import mediapipe as mp
import numpy as np
import keras
import math
from recognition_lib import data_processing as prc
from recognition_lib import model_detail as md


def twohand_mode(landmark1, landmark2):
    mode = False

    # 0,8 사이의 거리보다 중심점-중심점 거리가 작으면
    # 1손의 0,8 중심점 --- 2손의 0,8 중심점
    '''
    st1 = prc.get_distance(prc.point_to_list(landmark1[0]), prc.point_to_list(landmark1[12]))
    st2 = prc.get_distance(prc.point_to_list(landmark2[0]), prc.point_to_list(landmark2[12]))
    standard = ((st1+st2) / 2 ) * 1.5
    '''
    st1 = prc.get_distance(prc.point_to_list(landmark1[0]), prc.point_to_list(landmark1[9]))
    st2 = prc.get_distance(prc.point_to_list(landmark2[0]), prc.point_to_list(landmark2[9]))  
    standard = ((st1+st2) / 2 ) * 3

    m_pt1 = [(landmark1[0].x + landmark1[9].x)/2,
                (landmark1[0].y + landmark1[9].y)/2,
                (landmark1[0].z + landmark1[9].z)/2]
    m_pt2 = [(landmark2[0].x + landmark2[9].x)/2,
                (landmark2[0].y + landmark2[9].y)/2,
                (landmark2[0].z + landmark2[9].z)/2]
    distance = prc.get_distance(m_pt1, m_pt2)

    if distance <= standard:
        mode = True

    return distance, mode


def get_joint(landmark):
    joint = np.zeros((21, 3))
    joint[0] = [landmark[0].x, landmark[0].y, landmark[0].z]
    joint[1] = [landmark[1].x, landmark[1].y, landmark[1].z]
    joint[2] = [landmark[2].x, landmark[2].y, landmark[2].z]
    joint[3] = [landmark[3].x, landmark[3].y, landmark[3].z]
    joint[4] = [landmark[4].x, landmark[4].y, landmark[4].z]
    joint[5] = [landmark[5].x, landmark[5].y, landmark[5].z]
    joint[6] = [landmark[6].x, landmark[6].y, landmark[6].z]
    joint[7] = [landmark[7].x, landmark[7].y, landmark[7].z]
    joint[8] = [landmark[8].x, landmark[8].y, landmark[8].z]
    joint[9] = [landmark[9].x, landmark[9].y, landmark[9].z]
    joint[10] = [landmark[10].x, landmark[10].y, landmark[10].z]
    joint[11] = [landmark[11].x, landmark[11].y, landmark[11].z]
    joint[12] = [landmark[12].x, landmark[12].y, landmark[12].z]
    joint[13] = [landmark[13].x, landmark[13].y, landmark[13].z]
    joint[14] = [landmark[14].x, landmark[14].y, landmark[14].z]
    joint[15] = [landmark[15].x, landmark[15].y, landmark[15].z]
    joint[16] = [landmark[16].x, landmark[16].y, landmark[16].z]
    joint[17] = [landmark[17].x, landmark[17].y, landmark[17].z]
    joint[18] = [landmark[18].x, landmark[18].y, landmark[18].z]
    joint[19] = [landmark[19].x, landmark[19].y, landmark[19].z]
    joint[20] = [landmark[20].x, landmark[20].y, landmark[20].z]

    return joint


def get_data(landmark):

    joint = get_joint(landmark)
    v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:]
    v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:]
    v = v2 - v1

    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

    angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:]))
    angle = np.degrees(angle)

    data = np.array([angle], dtype=np.float32)

    return data


def onehand_model(landmark):

    data = get_data(landmark)    
    model = keras.models.load_model('ML/one_DNN_stop.h5')

    y_prob = model.predict(data, verbose=0)

    predicted = 14 #'normal state'
    for i in y_prob[0]:
        if i >= 0.9:
            predicted = y_prob.argmax(axis=-1)

    if predicted==0: # 'cat' class
        label = 1
        position = md.cat_mode(landmark)[1]
    elif predicted==1: # 'rock' class
        label = 9
        position = md.rock_mode(landmark)[1]
    elif predicted==2: # 'dog1' class
        label = 11
        position = md.dog1_mode(landmark)[1]
    elif predicted==3: # 'dog2' class
        label = 12
        position = md.dog2_mode(landmark)[1]
    elif predicted==4: # 'dog3' class
        label = 13
        position = md.dog3_mode(landmark)[1]
    elif predicted==5: # 'STOP' class
        label = 15
        position = [0, 0]
    else: # 'NORMAL'
        label = 14
        position = [0, 0]

    return label, position


def twohand_model(landmark1, landmark2):
    position=[0,0]

    data1 = get_data(landmark1)
    data2 = get_data(landmark2)
    data = np.concatenate((data1, data2), axis=1)   

    model = keras.models.load_model('ML/two_DNN_duck_flower2.h5')


    y_prob = model.predict(data, verbose=0)


    predicted = 14
    for i in y_prob[0]:
        if i >= 0.9:
            predicted = y_prob.argmax(axis=-1)

    if predicted==0 and md.butterfly_mode(landmark1, landmark2)[0]==True: # 'butterfly' class
        label = 0
        position = md.butterfly_mode(landmark1, landmark2)[1]

    elif predicted==1 and md.snail_mode(landmark1, landmark2)[0]==True: # 'snail' class
        label = 2
        position = md.snail_mode(landmark1, landmark2)[1]

    elif predicted==2 and md.deer_mode(landmark1, landmark2)[0]==True: # 'deer' class
        label = 3
        position = md.deer_mode(landmark1, landmark2)[1]

    elif predicted==3 and md.heart_mode(landmark1, landmark2)[0]==True: # 'heart' class
        label = 4
        position = md.heart_mode(landmark1, landmark2)[1]

    elif predicted==4 and md.duck_mode(landmark1, landmark2)[0]==True: # 'duck' class
        label = 5
        position = md.duck_mode(landmark1, landmark2)[1]

    elif predicted==5 and md.sun_mode(landmark1, landmark2)[0]==True: # 'sun' class
        label = 6
        position = md.sun_mode(landmark1, landmark2)[1]

    elif predicted==6 and md.house_mode(landmark1, landmark2)[0]==True: # 'house' class
        label = 7
        position = md.house_mode(landmark1, landmark2)[1]

    elif predicted==7 and md.tree_mode(landmark1, landmark2)[0]==True: # 'tree' class
        label = 8
        position = md.tree_mode(landmark1, landmark2)[1]

    elif predicted==8 and md.flower_mode(landmark1, landmark2)[0]==True: # 'flower' class
        label = 10
        position = md.flower_mode(landmark1, landmark2)[1]

    else: # 'NORMAL'
        label = 14
        position = (0, 0) 

    return label, position