from cv2 import BORDER_CONSTANT, copyMakeBorder
import numpy as np
import cv2

from outpart_lib import road as rd


#############first way ###############3
def FirstGo(x_c, y_c):
  random_go = np.random.randint(1,5)
  if random_go == 1:
    rp, banbok, way = rd.goksun1(x_c, y_c)
  if random_go == 2:
    rp, banbok, way = rd.goksun2(x_c, y_c)
  if random_go == 3:
    rp, banbok, way = rd.goksun3(x_c, y_c)
  if random_go == 4:
    rp, banbok, way = rd.goksun4(x_c, y_c)
  
  return rp, banbok, way


##########frame reset################
def go_first_frame (mv):
  TF = mv.get(cv2.CAP_PROP_FRAME_COUNT)
  CF = mv.get(cv2.CAP_PROP_POS_FRAMES)
  if TF == CF:
    mv.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
##############road##################
def goksun1 (xc,yc):
  r = []
  ban = 100
  yr = 0
  way = 0
  for i in range(0,ban):
    xr = xc+i
    yr = round(0.004*(xr-xc)**2)+yc
    r.append([xr, yr])
    if xr < xc:
      way = -1
    elif xr > xc:
      way = 1
      
      
  return r, ban, way

def goksun2 (xc,yc):
  r = []
  ban = 100
  way = 0
  yr = 0
  for i in range(0,ban):
    xr = xc+i
    yr = round(-0.004*(xr-xc)**2)+yc
    r.append([xr, yr])
    if xr < xc:
      way = -1
    elif xr > xc:
      way = 1
  return r, ban, way

def goksun3 (xc,yc):
  r = []
  ban = 100
  way = 0
  yr = 0
  for i in range(0,ban):
    xr = xc-i
    yr = round(0.004*(xr-xc)**2)+yc
    r.append([xr, yr])
    if xr < xc:
      way = -1
    elif xr > xc:
      way = 1
  return r, ban, way

def goksun4 (xc,yc):
  r = []
  ban = 100
  yr = 0
  way = 0
  for i in range(0,ban):
    xr = xc-i
    yr = round(-0.004*(xr-xc)**2)+yc
    r.append([xr, yr])
    if xr < xc:
      way = -1
    elif xr > xc:
      way = 1
  return r, ban, way


#-----wall---------#
def goksun1_1 (xc,yc):
  r = []
  ban = 50
  yr = 0
  way = 0
  for i in range(0,ban):
    xr = xc+i
    yr = yc
    r.append([xr, yr])
    if xr < xc:
      way = -1
    elif xr > xc:
      way = 1
  return r, ban, way

def goksun2_2 (xc,yc):
  r = []
  ban = 50
  yr = 0
  way = 0
  for i in range(0,ban):
    xr = xc
    yr = yc+i
    r.append([xr, yr])
    if xr < xc:
      way = -1
    elif xr > xc:
      way = 1
  return r, ban, way

def goksun3_3 (xc,yc):
  r = []
  ban = 50
  yr = 0
  way = 0
  for i in range(0,ban):
    xr = xc-i
    yr = yc
    r.append([xr, yr])
    if xr < xc:
      way = -1
    elif xr > xc:
      way = 1
  return r, ban, way

def goksun4_4 (xc,yc):
  r = []
  ban = 50
  yr = 0
  way = 0
  for i in range(0,ban):
    xr = xc
    yr = yc-i
    r.append([xr, yr])
    if xr < xc:
      way = -1
    elif xr > xc:
      way = 1
  return r, ban, way

###########Mask############
def masking (frame_resize, rp, c, bg, rows, cols):
  
  img_gray = cv2.cvtColor(frame_resize, cv2.COLOR_BGR2GRAY) 
  ret,img_mask = cv2.threshold(img_gray, 225, 255, cv2.THRESH_BINARY)
  img_mask_inv = cv2.bitwise_not(img_mask)
  img_roi = bg[rp[c][1]:rp[c][1]+rows, rp[c][0]:rp[c][0]+cols]
  img1 = cv2.bitwise_and(frame_resize, frame_resize, mask = img_mask_inv) 
  img2 = cv2.bitwise_and(img_roi, img_roi, mask=img_mask)
  dst = cv2.add(img1, img2)
  
  return dst

def s_masking (img, x, y, bg, rows, cols):
  
  img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
  ret,img_mask = cv2.threshold(img_gray, 220, 255, cv2.THRESH_BINARY)
  img_mask_inv = cv2.bitwise_not(img_mask)
  img_roi = bg[y:y+rows, x:x+cols]
  img1 = cv2.bitwise_and(img, img, mask = img_mask_inv) 
  img2 = cv2.bitwise_and(img_roi, img_roi, mask=img_mask)
  dst = cv2.add(img1, img2)
  
  return dst

def e_masking (img):
  src = cv2.rectangle(img, (0,0), (img.shape[1], img.shape[0]), (255,0,0), 3)
  return src