# common --------------------------------------------------------
from distutils.command.bdist import show_formats
import cv2
from cv2 import CAP_PROP_FRAME_WIDTH
from cv2 import CAP_PROP_FRAME_HEIGHT
import mediapipe as mp
import numpy as np
from classgo import Moving, noMoving

# outpart -------------------------------------------------------
import SbShSo as sss
from cmath import e
import random
import pyautogui

# recognition ---------------------------------------------------
from recognition_lib import util
from recognition_lib import MLmodel
from recognition_part import recognition
 
# for solving tensorflow error  ---------------------------------
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

# variable ------------------------------------------------------
nw, nh = pyautogui.size()
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0)
cap.set(CAP_PROP_FRAME_WIDTH, 1980);
cap.set(CAP_PROP_FRAME_HEIGHT, 1080);




s_btn = cv2.imread('outpart_lib/start_btn.png')
s_btn = cv2.resize(s_btn, (200,200))



s_bg = cv2.imread('outpart_lib/start_bg.jpg')
s_bg = cv2.resize(s_bg, (nw,nh))
s_btn = sss.s_masking(s_btn, 100, 100 , s_bg, 200,200)
s_bg[100:300, 100:300] = s_btn

ti = cv2.imread('outpart_lib/tutorial.png')
ti = cv2.resize(ti, (100, 100))
bg = cv2.imread('outpart_lib/bg3.jpg')
bg = cv2.resize(bg, (nw,nh))
bg[100:200,100:200] = ti
bg_clone = cv2.imread('outpart_lib/bg3.jpg')
bg_clone = cv2.resize(bg, (nw,nh))
bg_clone[100:200,100:200] = ti

nabi_1 = cv2.VideoCapture('outpart_lib/butterfly_1.gif')
nabi_2 = cv2.VideoCapture('outpart_lib/butterfly_3.gif')
nabi_3 = cv2.VideoCapture('outpart_lib/butterfly_4.gif')

cat_1 = cv2.VideoCapture('outpart_lib/cat_1.gif')
cat_2 = cv2.VideoCapture('outpart_lib/cat_2.gif')
cat_3 = cv2.VideoCapture('outpart_lib/cat_3.gif')
####################여기부터######################################
snail_1 = cv2.VideoCapture('outpart_lib/snail_1.gif')
snail_2 = cv2.VideoCapture('outpart_lib/snail_2.gif')
snail_3 = cv2.VideoCapture('outpart_lib/snail_3.gif')

deer_1 = cv2.VideoCapture('outpart_lib/deer.gif')
deer_2 = cv2.VideoCapture('outpart_lib/deer_1.gif')
deer_3 = cv2.VideoCapture('outpart_lib/deer_2.gif')

heart_1 = cv2.VideoCapture('outpart_lib/heart_1.gif')
heart_2 = cv2.VideoCapture('outpart_lib/heart_1.gif')
heart_3 = cv2.VideoCapture('outpart_lib/heart_1.gif')

duck_1 = cv2.VideoCapture('outpart_lib/duck_1.gif')
duck_2 = cv2.VideoCapture('outpart_lib/duck_2.gif')
duck_3 = cv2.VideoCapture('outpart_lib/duck_3.gif')

sun_1 = cv2.VideoCapture('outpart_lib/sun.gif')
sun_2 = cv2.VideoCapture('outpart_lib/sun.gif')
sun_3 = cv2.VideoCapture('outpart_lib/sun.gif')

house_1 = cv2.VideoCapture('outpart_lib/house.gif')
house_2 = cv2.VideoCapture('outpart_lib/house.gif')
house_3 = cv2.VideoCapture('outpart_lib/house.gif')

tree_1 = cv2.VideoCapture('outpart_lib/tree.gif')
tree_2 = cv2.VideoCapture('outpart_lib/tree.gif')
tree_3 = cv2.VideoCapture('outpart_lib/tree.gif')

rock_1 = cv2.VideoCapture('outpart_lib/rock.gif')
rock_2 = cv2.VideoCapture('outpart_lib/rock.gif')
rock_3 = cv2.VideoCapture('outpart_lib/rock.gif')

flower_1 = cv2.VideoCapture('outpart_lib/flower.gif')
flower_2 = cv2.VideoCapture('outpart_lib/flower.gif')
flower_3 = cv2.VideoCapture('outpart_lib/flower.gif')

dog_1 = cv2.VideoCapture('outpart_lib/dog_1.gif')
dog_2 = cv2.VideoCapture('outpart_lib/dog_3.gif')
dog_3 = cv2.VideoCapture('outpart_lib/dog_7.gif')
################################여기까지 이미지 입력#############################

tuto = cv2.imread('outpart_lib/tutorial_img.png')
tuto = cv2.resize(tuto, (700, 700))
ui = cv2.imread('outpart_lib/back.png')
ui = cv2.resize(ui, (210, 210))
ji = cv2.imread('outpart_lib/eraser.png')
ji = cv2.resize(ji,(150,150))
bi = cv2.imread('outpart_lib/back.png')
bi = cv2.resize(bi,(150,150))
ei = cv2.imread('outpart_lib/off.png')
ei = cv2.resize(ei, (150,150))
s, g, ch = bg.shape


ediya = []

shibal = -1
flag = []
static_flag = []
make_count = 0
tmpc=[]
detect_label = []
state_flag = -1
moving_switch = True
die = False
tutorial = 0
#show_type = 0

s1 = sss.goksun1
s2= sss.goksun2
s3= sss.goksun3
s4= sss.goksun4

s1_1 = sss.goksun1_1
s2_2= sss.goksun2_2
s3_3= sss.goksun3_3
s4_4= sss.goksun4_4
def s_onMouse(event, x,y, flags, praram):
    global s_bg
    global state_flag
    if event == cv2.EVENT_LBUTTONDOWN:
        if (x>100 and x<300) and (y>100 and y<300):
            state_flag = 0
            
def onMouse(event, x,y, flags, praram):
    global tutorial
    if event == cv2.EVENT_LBUTTONDOWN:
        if ((x>100) and (x<200)) and ((y>100) and (y<200)):
            print('0000000000000000000')
            tutorial += 1
            
def mouse_event(event, x, y, flags, param):
    global show_type
    global state_flag
    global die
    if event == cv2.EVENT_LBUTTONDOWN:
        if  ((x>100)and(x<250)) and ((y>100)and(y<250)):
            state_flag = 0
        if  ((x>1500)and(x<1650)) and ((y>100)and(y<250)):
            state_flag = 2
        if  ((x>1700)and(x<1850)) and ((y>100)and(y<250)):
            die = True
        
            
def erase_event(event, x, y, flags, param):
                    global state_flag
                    global bg
                    global erase_bg
                    global flag
                    global static_flag
                    if event == cv2.EVENT_LBUTTONDOWN:
                        for j in flag:
                            if  ((x>j.rp[j.c][0])and(x<j.rp[j.c][0]+j.col)) and ((y>j.rp[j.c][1])and(y<j.rp[j.c][1]+j.row)):
                                erase_bg[j.rp[j.c][1]:j.rp[j.c][1]+j.row,\
                                    j.rp[j.c][0]:j.rp[j.c][0]+j.col] = j.RF
                                flag.remove(j)    
                        for k in static_flag:  
                            if  ((x>k.sx)and(x<k.sx+k.scol)) and ((y>k.sy)and(y<k.sy+k.srow)):
                                print('aaaaaaaaaaaaaaaa')
                                erase_bg[k.sy:k.sy+k.srow, k.sx: k.sx+k.scol] = k.sRF
                                #bg_clone[j.sx: j.sx+j.scol, j.sy:j.sy+j.srow] = j.sRF
                                static_flag.remove(k)
                              
                        if  ((x>1500)and(x<1650)) and ((y>100)and(y<250)):
                            state_flag = 0
                            bg[100:100+bi.shape[0], 1500:1500+bi.shape[1]] = bg_clone[100:100+bi.shape[0], 1500:1500+bi.shape[1]]
                    
                            #print(state_flag)                

# code stat -----------------------------------------------------
with mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=6,
    min_detection_confidence=0.5,
    ) as hands:
    
    counting = 1 # 인식파트에서 사용하는 변수
    dict_hand = util.make_dict_hand() # 인식파트에서 사용하는 변수

    while cv2.waitKey(10)!=ord('q'):
        #cv2.imshow('eee',bg_clone)
        if die:
            break
        
        success, image = cap.read()
        
        if not success:
            continue

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.resize(image,(nw,nh))



        ############################### 인식 파트 시작 ###############################
        if state_flag == 0:
            if results.multi_hand_landmarks:
                recog = recognition(image, results.multi_hand_landmarks, dict_hand, counting)
                recog.draw_load_hand()

                if (counting%10==0):
                    detect_label = recog.recog_main()
                    #print(detect_label)
                
                    if detect_label:    
                        for i in range(0, len(detect_label)):
        #####################움직여#########################################                    
                            if detect_label[i][0] == 0:
                                rndN = np.random.randint(1, 4)
                                if rndN == 1:
                                    nabi = nabi_1
                                elif rndN == 2:
                                    nabi = nabi_2
                                elif rndN == 3:
                                    nabi = nabi_3
                                x_c = int((detect_label[i][1][0])*g)
                                y_c = int((detect_label[i][1][1])*s)
                                if (x_c+100 > nw) or (y_c+100>nh):
                                    x_c = nw - 100
                                    y_c = nh - 100
                                print(x_c)
                                print(y_c)
                                tmpc.append([x_c,y_c])
                                make_count+=1
                                for _ in range(0,make_count):
                                    if shibal < _ :
                                        globals()['nabi'+str(_)] = Moving(nabi,tmpc[0][0], tmpc[0][1], 100, 100, 1)
                                        flag.append(globals()['nabi'+str(_)])
                                        shibal+=1
                                        del tmpc[0]
                            
                            elif detect_label[i][0] == 1:
                                rndN = np.random.randint(1, 4)
                                if rndN == 1:
                                    cat = cat_1
                                elif rndN == 2:
                                    cat = cat_2
                                elif rndN == 3:
                                    cat = cat_3
                                x_c = int((detect_label[i][1][0])*g)
                                y_c = int((detect_label[i][1][1])*s)
                                if (x_c+200 > nw) or (y_c+200>nh):
                                    x_c = nw - 200
                                    y_c = nh - 200
                                print(x_c)
                                print(y_c)
                                tmpc.append([x_c,y_c])
                                make_count+=1
                                for _ in range(0,make_count):
                                    if shibal < _ :
                                        globals()['cat'+str(_)] = Moving(cat,tmpc[0][0], tmpc[0][1], 200, 200, 1)
                                        flag.append(globals()['cat'+str(_)])
                                        shibal+=1
                                        del tmpc[0]
                            
                            elif detect_label[i][0] == 2:
                                rndN = np.random.randint(1, 4)
                                if rndN == 1:
                                    snail = snail_1
                                elif rndN == 2:
                                    snail = snail_2
                                elif rndN == 3:
                                    snail = snail_3
                                x_c = int((detect_label[i][1][0])*g)
                                y_c = int((detect_label[i][1][1])*s)
                                if (x_c+70 > nw) or (y_c+70>nh):
                                    x_c = nw - 70
                                    y_c = nh - 70
                                print(x_c)
                                print(y_c)
                                tmpc.append([x_c,y_c])
                                make_count+=1
                                for _ in range(0,make_count):
                                    if shibal < _ :
                                        globals()['snail'+str(_)] = Moving(snail,tmpc[0][0], tmpc[0][1], 70, 70, 1)
                                        flag.append(globals()['snail'+str(_)])
                                        shibal+=1
                                        del tmpc[0]
                            
                            elif detect_label[i][0] == 3:
                                rndN = np.random.randint(1, 4)
                                if rndN == 1:
                                    deer = deer_1
                                elif rndN == 2:
                                    deer = deer_2
                                elif rndN == 3:
                                    deer = deer_3
                                x_c = int((detect_label[i][1][0])*g)
                                y_c = int((detect_label[i][1][1])*s)
                                if (x_c+300 > nw) or (y_c+300>nh):
                                    x_c = nw - 300
                                    y_c = nh - 300
                                print(x_c)
                                print(y_c)
                                tmpc.append([x_c,y_c])
                                make_count+=1
                                for _ in range(0,make_count):
                                    if shibal < _ :
                                        globals()['deer'+str(_)] = Moving(deer,tmpc[0][0], tmpc[0][1], 300, 300, 1)
                                        flag.append(globals()['deer'+str(_)])
                                        shibal+=1
                                        del tmpc[0]    
                                        
                            elif detect_label[i][0] == 4:
                                rndN = np.random.randint(1, 4)
                                if rndN == 1:
                                    heart = heart_1
                                elif rndN == 2:
                                    heart = heart_2
                                elif rndN == 3:
                                    heart = heart_3
                                x_c = int((detect_label[i][1][0])*g)
                                y_c = int((detect_label[i][1][1])*s)
                                if (x_c+100 > nw) or (y_c+100>nh):
                                    x_c = nw - 100
                                    y_c = nh - 100
                                print(x_c)
                                print(y_c)
                                tmpc.append([x_c,y_c])
                                make_count+=1
                                for _ in range(0,make_count):
                                    if shibal < _ :
                                        globals()['ha'+str(_)] = Moving(heart,tmpc[0][0], tmpc[0][1], 100,100, 0)
                                        flag.append(globals()['ha'+str(_)])
                                        shibal+=1
                                        del tmpc[0]
                            
                            elif detect_label[i][0] == 5:
                                rndN = np.random.randint(1, 4)
                                if rndN == 1:
                                    duck = duck_1
                                elif rndN == 2:
                                    duck = duck_2
                                elif rndN == 3:
                                    duck = duck_3
                                x_c = int((detect_label[i][1][0])*g)
                                y_c = int((detect_label[i][1][1])*s)
                                if (x_c+150 > nw) or (y_c+150>nh):
                                    x_c = nw - 150
                                    y_c = nh - 150
                                print(x_c)
                                print(y_c)
                                tmpc.append([x_c,y_c])
                                make_count+=1
                                for _ in range(0,make_count):
                                    if shibal < _ :
                                        globals()['du'+str(_)] = Moving(duck,tmpc[0][0], tmpc[0][1], 150, 150, 1)
                                        flag.append(globals()['du'+str(_)])
                                        shibal+=1
                                        del tmpc[0]
                            
                            elif detect_label[i][0] == 6:
                                rndN = np.random.randint(1, 4)
                                if rndN == 1:
                                    sun = sun_1
                                elif rndN == 2:
                                    sun = sun_2
                                elif rndN == 3:
                                    sun = sun_3
                                x_c = int((detect_label[i][1][0])*g)
                                y_c = int((detect_label[i][1][1])*s)
                                if (x_c+180 > nw) or (y_c+180>nh):
                                    x_c = nw - 180
                                    y_c = nh - 180
                                print(x_c)
                                print(y_c)
                                tmpc.append([x_c,y_c])
                                make_count+=1
                                for _ in range(0,make_count):
                                    if shibal < _ :
                                        globals()['su'+str(_)] = Moving(sun,tmpc[0][0], tmpc[0][1], 180, 180, 0)
                                        flag.append(globals()['su'+str(_)])
                                        shibal+=1
                                        del tmpc[0]
                            elif detect_label[i][0] == 7:
                                rndN = np.random.randint(1, 4)
                                if rndN == 1:
                                    house = house_1
                                elif rndN == 2:
                                    house = house_2
                                elif rndN == 3:
                                    house = house_3
                                x_c = int((detect_label[i][1][0])*g)
                                y_c = int((detect_label[i][1][1])*s)-50
                                if (x_c+400 > nw) or (y_c+400>nh) :
                                    x_c = int(nw/2)
                                    y_c = int(nh/2)
                                
                                print(x_c)
                                print(y_c)
                                tmpc.append([x_c,y_c])
                                make_count+=1
                                for _ in range(0,make_count):
                                    if shibal < _ :
                                        globals()['hau'+str(_)] = Moving(house,tmpc[0][0], tmpc[0][1], 400, 400, 0)
                                        flag.append(globals()['hau'+str(_)])
                                        shibal+=1
                                        del tmpc[0]
                                        
                            elif detect_label[i][0] == 8:
                                rndN = np.random.randint(1, 4)
                                if rndN == 1:
                                    tree = tree_1
                                elif rndN == 2:
                                    tree = tree_2
                                elif rndN == 3:
                                    tree = tree_3
                                x_c = int((detect_label[i][1][0])*g)
                                y_c = int((detect_label[i][1][1])*s)
                                if (x_c+300 > nw) or (y_c+300>nh):
                                    x_c = nw - 300
                                    y_c = nh - 300
                                print(x_c)
                                print(y_c)
                                tmpc.append([x_c,y_c])
                                make_count+=1
                                for _ in range(0,make_count):
                                    if shibal < _ :
                                        globals()['na'+str(_)] = Moving(tree,tmpc[0][0], tmpc[0][1], 300, 300, 0)
                                        flag.append(globals()['na'+str(_)])
                                        shibal+=1
                                        del tmpc[0]
                            
                            elif detect_label[i][0] == 9:
                                rndN = np.random.randint(1, 4)
                                if rndN == 1:
                                    rock = rock_1
                                elif rndN == 2:
                                    rock = rock_2
                                elif rndN == 3:
                                    rock = rock_3
                                x_c = int((detect_label[i][1][0])*g)
                                y_c = int((detect_label[i][1][1])*s)
                                if (x_c+100 > nw) or (y_c+100>nh):
                                    x_c = nw - 100
                                    y_c = nh - 100
                                print(x_c)
                                print(y_c)
                                tmpc.append([x_c,y_c])
                                make_count+=1
                                for _ in range(0,make_count):
                                    if shibal < _ :
                                        globals()['ro'+str(_)] = Moving(rock,tmpc[0][0], tmpc[0][1], 100, 100, 0)
                                        flag.append(globals()['ro'+str(_)])
                                        shibal+=1
                                        del tmpc[0]
                            
                            elif detect_label[i][0] == 10:
                                rndN = np.random.randint(1, 4)
                                if rndN == 1:
                                    flower = flower_1
                                elif rndN == 2:
                                    flower = flower_2
                                elif rndN == 3:
                                    flower = flower_3
                                x_c = int((detect_label[i][1][0])*g)
                                y_c = int((detect_label[i][1][1])*s)
                                if (x_c+100 > nw) or (y_c+100>nh):
                                    x_c = nw - 100
                                    y_c = nh - 100
                                print(x_c)
                                print(y_c)
                                tmpc.append([x_c,y_c])
                                make_count+=1
                                for _ in range(0,make_count):
                                    if shibal < _ :
                                        globals()['fl'+str(_)] = Moving(flower,tmpc[0][0], tmpc[0][1], 100, 100, 0)
                                        flag.append(globals()['fl'+str(_)])
                                        shibal+=1
                                        del tmpc[0]
                            
                            elif detect_label[i][0] == 11 or detect_label[i][0] == 12 or detect_label[i][0] == 13:
                                print(detect_label)
                                rndN = np.random.randint(1, 4)
                                if rndN == 1:
                                    dog = dog_1
                                elif rndN == 2:
                                    dog = dog_2
                                elif rndN == 3:
                                    dog = dog_3
                                x_c = int((detect_label[i][1][0][0])*g)
                                y_c = int((detect_label[i][1][0][1])*s)
                                if (x_c+200 > nw) or (y_c+200>nh):
                                    x_c = nw - 200
                                    y_c = nh - 200
                                print(x_c)
                                print(y_c)
                                tmpc.append([x_c,y_c])
                                make_count+=1
                                for _ in range(0,make_count):
                                    if shibal < _ :
                                        globals()['do'+str(_)] = Moving(dog,tmpc[0][0], tmpc[0][1], 200, 200, 1)
                                        flag.append(globals()['do'+str(_)])
                                        shibal+=1
                                        del tmpc[0]
                            
                            elif detect_label[i][0] == 15:
                                state_flag = 1
                                
                            
                            
                            
        counting += 1
        ###송출###
        
        if state_flag == -1:
            show_type = -1
        elif state_flag == 0:
            moving_switch = True
            show_type = 0
        elif state_flag == 1:
            moving_switch = False
            show_type = 1
        elif state_flag == 2:
            moving_switch = False
            show_type = 2
        
        
        if flag:
            for i in flag:
                if moving_switch == True:
                    i.frame = i.gif.read()[1]
                
                    if i.way == -1:    
                        i.frame = cv2.resize(i.frame, (i.resize_x, i.resize_y))
                    elif i.way == 1:
                        i.frame = cv2.resize(i.frame, (i.resize_x, i.resize_y))
                        i.frame = cv2.flip(i.frame, 1)
                i.row = i.frame.shape[0]
                i.col = i.frame.shape[1]
                sss.go_first_frame(i.gif)
                i.RF = bg_clone[i.rp[i.c][1]:i.rp[i.c][1]+i.row,\
                    i.rp[i.c][0]:i.rp[i.c][0]+i.col]
                if (state_flag == 0) or (state_flag == 1): 
                    i.dst = sss.masking(i.frame, i.rp, i.c, bg, i.row, i.col)
                    #i.dst = i.frame
                elif state_flag == 2:
                    i.dst = sss.masking(i.frame, i.rp, i.c, bg, i.row, i.col)
                    i.dst = sss.e_masking(i.dst)
                    
                bg[i.rp[i.c][1]:i.rp[i.c][1]+i.row,\
                    i.rp[i.c][0]:i.rp[i.c][0]+i.col] = i.dst
                
                stop_bg = bg
                erase_bg = bg

        if show_type == -1:
            cv2.imshow('test', s_bg)
            cv2.setMouseCallback("test",s_onMouse,s_bg)
            

        elif show_type == 0:
            if tutorial%2 == 1:
              bg[200:200+tuto.shape[0], 200:200+tuto.shape[1]] = tuto
            #cv2.imshow('rerere',tutobg)
            cv2.imshow('test', bg)
            cv2.setMouseCallback("test",onMouse,bg)
        elif show_type == 1:
            stop_bg = cv2.GaussianBlur(bg,(5,5),e)
            ui = sss.s_masking(ui, 60,45,stop_bg,ui.shape[0],ui.shape[1])
            stop_bg[45:45+ui.shape[0], 60:60+ui.shape[1]] = ui
            ji = sss.s_masking(ji, 1500,100,stop_bg,ji.shape[0],ji.shape[1])
            stop_bg[100:100+ji.shape[0], 1500:1500+ji.shape[1]] = ji
            ei = sss.s_masking(ei, 1700,100,stop_bg,ei.shape[0],ei.shape[1])
            stop_bg[100:100+ei.shape[0], 1700:1700+ei.shape[1]] = ei
            cv2.imshow('test', stop_bg)
            cv2.setMouseCallback("test",mouse_event,stop_bg)
        elif show_type == 2:
            #bi_RF = bg[100:100+bi.shape[1], 100:100+bi.shape[0]]
            bi = sss.s_masking(bi, 1500,100,bg,bi.shape[0],bi.shape[1])
            bg[100:100+bi.shape[0], 1500:1500+bi.shape[1]] = bi
            cv2.imshow('test', bg)
            cv2.setMouseCallback("test",erase_event,bg)

            

            
        
        if tutorial%2 == 0:
            bg[200:200+tuto.shape[0], 200:200+tuto.shape[1]] = bg_clone[200:200+tuto.shape[0], 200:200+tuto.shape[1]] 
    
        if flag:        
            if moving_switch == True:
                for i in flag:    
                    bg[i.rp[i.c][1]:i.rp[i.c][1]+i.row,\
                        i.rp[i.c][0]:i.rp[i.c][0]+i.col] = i.RF
                       

                    
                    if (i.c+1)==i.banbok:
                        option = np.random.randint(1, 5)
                        if option == 1:
                            i.rp, i.banbok, i.way =s1(i.rp[i.c][0],i.rp[i.c][1])
                        elif option == 2:
                            i.rp, i.banbok, i.way =s2(i.rp[i.c][0],i.rp[i.c][1])
                        elif option == 3:
                            i.rp, i.banbok, i.way =s3(i.rp[i.c][0],i.rp[i.c][1])
                        elif option == 4:
                            i.rp, i.banbok, i.way =s4(i.rp[i.c][0],i.rp[i.c][1])
                        i.c = 0
                        
                    if (i.rp[i.c][0] == 0) :
                        i.rp, i.banbok, i.way = s1_1(i.rp[i.c][0],i.rp[i.c][1])
                        i.c = 0
                    elif (i.rp[i.c][1] == 0):
                        i.rp, i.banbok, i.way = s2_2(i.rp[i.c][0],i.rp[i.c][1])
                        i.c = 0
                    elif (i.rp[i.c][0]+i.col == g):
                        i.rp, i.banbok, i.way = s3_3(i.rp[i.c][0],i.rp[i.c][1])
                        i.c = 0
                    elif (i.rp[i.c][1]+i.row == s):
                        i.rp, i.banbok, i.way = s4_4(i.rp[i.c][0],i.rp[i.c][1])
                        i.c = 0   
                        
                    if i.jord == 1:
                        i.c+=1
        
        
        if tutorial == 10:
            tutorial = 0

        

##############################################################################

        #image.resize(image,(nh,nw))
        cv2.imshow('Capstone_test', image)

cv2.destroyAllWindows()
cap.release()