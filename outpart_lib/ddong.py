import cv2
import mediapipe as mp
import numpy as np
from cmath import e
import random

from util import spread_angle, spread_state, fold_angle, fold_state, input_name
from util import hand_shape
import SbShSo as sss

 
 
bg = cv2.imread('bg3.jpg')
bg_clone = cv2.imread('bg3.jpg')

gif2 = cv2.VideoCapture('butter.gif')
ui = cv2.imread('close.png')
s, g, ch = bg.shape
######정적이미지######
haha = cv2.imread('tree.png')
static_img = cv2.resize(haha, (500,500))

x_c = 400
y_c = 100



banbokn0=0
name = []
a=0
shape_label = 0
f=0
t=True
ship=0
shibal = -1

option =0
 
 
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
 
cap = cv2.VideoCapture(0)
 
with mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    
    counting = 0
    while cv2.waitKey(10)!=ord('q'):
        cv2.imshow('test', bg)
        success, image = cap.read()
        
        if not success:
            continue
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
 
        results = hands.process(image)
 
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cap_width = cap.get(3)
        cap_height = cap.get(4)

        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:

                if (counting%2) == 0:
                    hand_1 = hand.landmark
                    spread_angle1 = spread_angle(hand_1)
                    spread_state1 = spread_state(spread_angle1)
                    fold_angle1 = fold_angle(hand_1)
                    fold_state1 = fold_state(fold_angle1)
                    
                if (counting%2) == 1:
                    hand_2 = hand.landmark
                    spread_angle2 = spread_angle(hand_2)
                    spread_state2 = spread_state(spread_angle2)
                    fold_angle2 = fold_angle(hand_2)
                    fold_state2 = fold_state(fold_angle2)

                if counting>0:
                    shape_label, shape_point = hand_shape(spread_angle1, spread_angle2, spread_state1, spread_state2, 
                                                            fold_angle1, fold_angle2, fold_state1, fold_state2, hand_1, hand_2)
                    if shape_label > 0:
                        ship = shape_label
                        if ship==1:
                            a+=1
                    
                    
                    input_name(shape_label, image)

                counting += 1

                # 뼈대 그려주기
                mp_drawing.draw_landmarks(
                    image, hand, mp_hands.HAND_CONNECTIONS)

        if a>0:
            
            for i in range(0,a):
                    if shibal<i:
                        
                        globals()['gifn'+str(i)] = cv2.VideoCapture('ddd.gif')
                        globals()['rpn'+str(i)],  globals()['banbokn'+str(i)],  globals()['wayn'+str(i)]\
                                    = sss.FirstGo( x_c, y_c)
                        globals()['cn'+str(i)]=0

                        shibal+=1

            for i in range (0, a):
                    
                    sss.go_first_frame(globals()['gifn'+str(i)])
                        
                    globals()['framen'+str(i)] = globals()['gifn'+str(i)].read()[1]
                    globals()['frame_resizen'+str(i)] = cv2.resize(globals()['framen'+str(i)], (150, 150))
                    globals()['rowsn'+str(i)] = globals()['frame_resizen'+str(i)].shape[0]
                    globals()['closn'+str(i)] = globals()['frame_resizen'+str(i)].shape[1]
            
            for i in range (0, a):
                    
                    globals()['RFn'+str(i)] = bg_clone[globals()['rpn'+str(i)][globals()['cn'+str(i)]][1]:\
                                                        globals()['rpn'+str(i)][globals()['cn'+str(i)]][1]+globals()['rowsn'+str(i)], \
                                                        globals()['rpn'+str(i)][globals()['cn'+str(i)]][0]:\
                                                        globals()['rpn'+str(i)][globals()['cn'+str(i)]][0]+globals()['closn'+str(i)]]
            
            for i in range (0, a):        
                    globals()['dstn'+str(i)] = sss.masking(globals()['frame_resizen'+str(i)], globals()['rpn'+str(i)], \
                                                                    globals()['cn'+str(i)], bg, globals()['rowsn'+str(i)], globals()['closn'+str(i)] )
                    bg[globals()['rpn'+str(i)][globals()['cn'+str(i)]][1]:\
                                                        globals()['rpn'+str(i)][globals()['cn'+str(i)]][1]+globals()['rowsn'+str(i)], \
                                                        globals()['rpn'+str(i)][globals()['cn'+str(i)]][0]:\
                                                        globals()['rpn'+str(i)][globals()['cn'+str(i)]][0]+globals()['closn'+str(i)]]\
                                                            = globals()['dstn'+str(i)]
            
            for i in range (0, a):            
                    cv2.imshow('test',bg)
                    
            for i in range (0, a):        
                    bg[globals()['rpn'+str(i)][globals()['cn'+str(i)]][1]:\
                                                        globals()['rpn'+str(i)][globals()['cn'+str(i)]][1]+globals()['rowsn'+str(i)], \
                                                        globals()['rpn'+str(i)][globals()['cn'+str(i)]][0]:\
                                                        globals()['rpn'+str(i)][globals()['cn'+str(i)]][0]+globals()['closn'+str(i)]]\
                                                            = globals()['RFn'+str(i)]
                                                        
                                                        
            for i in range (0, a):                 
                    if (globals()['cn'+str(i)]+1)==globals()['banbokn'+str(i)]:
                        option = np.random.randint(1, 5)
                        print('dhdhdh  %d' % option)
                        if option == 1:
                            globals()['rpn'+str(i)]=sss.goksun1(globals()['rpn'+str(i)][globals()['cn'+str(i)]][0],globals()['rpn'+str(i)][globals()['cn'+str(i)]][1])[0]
                            globals()['banbokn'+str(i)]=sss.goksun1(globals()['rpn'+str(i)][globals()['cn'+str(i)]][0],globals()['rpn'+str(i)][globals()['cn'+str(i)]][1])[1]
                            globals()['wayn'+str(i)]=sss.goksun1(globals()['rpn'+str(i)][globals()['cn'+str(i)]][0],globals()['rpn'+str(i)][globals()['cn'+str(i)]][1])[2]
                        elif option == 2:
                            globals()['rpn'+str(i)]=sss.goksun2(globals()['rpn'+str(i)][globals()['cn'+str(i)]][0],globals()['rpn'+str(i)][globals()['cn'+str(i)]][1])[0]
                            globals()['banbokn'+str(i)]=sss.goksun2(globals()['rpn'+str(i)][globals()['cn'+str(i)]][0],globals()['rpn'+str(i)][globals()['cn'+str(i)]][1])[1]
                            globals()['wayn'+str(i)]=sss.goksun2(globals()['rpn'+str(i)][globals()['cn'+str(i)]][0],globals()['rpn'+str(i)][globals()['cn'+str(i)]][1])[2]
                        elif option == 3:
                            globals()['rpn'+str(i)]=sss.goksun3(globals()['rpn'+str(i)][globals()['cn'+str(i)]][0],globals()['rpn'+str(i)][globals()['cn'+str(i)]][1])[0]
                            globals()['banbokn'+str(i)]=sss.goksun3(globals()['rpn'+str(i)][globals()['cn'+str(i)]][0],globals()['rpn'+str(i)][globals()['cn'+str(i)]][1])[1]
                            globals()['wayn'+str(i)]=sss.goksun3(globals()['rpn'+str(i)][globals()['cn'+str(i)]][0],globals()['rpn'+str(i)][globals()['cn'+str(i)]][1])[2]
                        elif option == 4:
                            globals()['rpn'+str(i)]=sss.goksun4(globals()['rpn'+str(i)][globals()['cn'+str(i)]][0],globals()['rpn'+str(i)][globals()['cn'+str(i)]][1])[0]
                            globals()['banbokn'+str(i)]=sss.goksun4(globals()['rpn'+str(i)][globals()['cn'+str(i)]][0],globals()['rpn'+str(i)][globals()['cn'+str(i)]][1])[1]
                            globals()['wayn'+str(i)]=sss.goksun4(globals()['rpn'+str(i)][globals()['cn'+str(i)]][0],globals()['rpn'+str(i)][globals()['cn'+str(i)]][1])[2]

                        globals()['cn'+str(i)] = 0

       
                    if (globals()['rpn'+str(i)][globals()['cn'+str(i)]][0] == 0) :
                        
                        globals()['rpn'+str(i)], globals()['banbokn'+str(i)], globals()['wayn'+str(i)] \
                            = sss.goksun1_1(globals()['rpn'+str(i)][globals()['cn'+str(i)]][0],globals()['rpn'+str(i)][globals()['cn'+str(i)]][1])
                        globals()['cn'+str(i)]=0
                        print('aaa')
                    
                    elif (globals()['rpn'+str(i)][globals()['cn'+str(i)]][1] == 0):
                        globals()['rpn'+str(i)], globals()['banbokn'+str(i)], globals()['wayn'+str(i)] \
                            = sss.goksun2_2(globals()['rpn'+str(i)][globals()['cn'+str(i)]][0],globals()['rpn'+str(i)][globals()['cn'+str(i)]][1])
                        
                        globals()['cn'+str(i)]=0
                        print('bbb')
                
                    elif (globals()['rpn'+str(i)][globals()['cn'+str(i)]][0]+globals()['closn'+str(i)] == g):
                        globals()['rpn'+str(i)], globals()['banbokn'+str(i)], globals()['wayn'+str(i)] \
                            = sss.goksun3_3(globals()['rpn'+str(i)][globals()['cn'+str(i)]][0],globals()['rpn'+str(i)][globals()['cn'+str(i)]][1])
                        globals()['cn'+str(i)]=0
                        print('ccc')

                    elif (globals()['rpn'+str(i)][globals()['cn'+str(i)]][1]+globals()['rowsn'+str(i)] == s):
                        globals()['rpn'+str(i)], globals()['banbokn'+str(i)], globals()['wayn'+str(i)] \
                            = sss.goksun4_4(globals()['rpn'+str(i)][globals()['cn'+str(i)]][0],globals()['rpn'+str(i)][globals()['cn'+str(i)]][1])
                        globals()['cn'+str(i)]=0   
                        print('ddd')
                    
        

                    globals()['cn'+str(i)]+=1
                        
                
                        
                    
                    
                    ###################################################
                    
                    
                    

                    
                    
                    ####################################################
            
                
                
                #print(str(globals()['rpn'+str(i)])) #객체 하나 출력 사용 기능
        print('xxxxxxxxxxxx') 
        #print('a=%d' % a)
        #print('shi=%d' % shibal)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        cv2.imshow('Capstone_test', image)
        if cv2.waitKey(1) == ord('q'):
            break
 
cap.release()