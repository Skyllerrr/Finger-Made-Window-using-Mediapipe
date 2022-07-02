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


###########3
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