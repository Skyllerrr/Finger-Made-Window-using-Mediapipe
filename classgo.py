import cv2
import SbShSo as sss


class Moving :

  gif = []
  rp = []
  banbok = 0
  way = 0
  c = 0
  frame = 0
  row = 0
  col = 0
  RF = []
  dst = []
  x = 0
  y = 0
  RF = []
  resize_x = 0
  resize_y = 0
  jord = 0
  
  def __init__(self, gif, x,y, resize_x, resize_y, jord):
    self.rp, self.banbok, self.way = sss.FirstGo(x,y)
    self.gif = gif
    self.resize_x = resize_x
    self.resize_y = resize_y
    self.jord = jord
    
class noMoving : 
  simg = []
  sx, sy = 0, 0
  srow, scol, sch = 0,0,0
  sRF = []
  sdst = []
  
  def __init__(self, img, x, y, RF):
    self.simg = img
    self.sRF = RF
    self.sx, self.sy = x , y
    self.srow, self.scol, self.sch = self.simg.shape
    
    
    


    
    
    
  
  
  
  
  