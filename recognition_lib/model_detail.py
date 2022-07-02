from turtle import position
from recognition_lib import data_processing as prc


# ONEHAND detail -------------------------------------------
def cat_mode(landmark):
    class_mode = False

    s_angle = prc.spread_angle(landmark)
    state = prc.spread_state(s_angle)

    if state[1]==1 and state[2]==1:
        class_mode = True
    position = [landmark[8].x, landmark[8].y]

    return class_mode, position

def rock_mode(landmark):
    class_mode = False

    s_angle = prc.spread_angle(landmark)
    state = prc.spread_state(s_angle)

    if state==[1,1,1,1]:
        class_mode = True
    position = [landmark[8].x, landmark[8].y]

    return class_mode, position

def dog1_mode(landmark):
    class_mode = False

    s_angle = prc.spread_angle(landmark)
    state = prc.spread_state(s_angle)

    if state==[1,1,0,0]:
        class_mode = True
    position = [ [landmark[8].x, landmark[8].y] ]

    return class_mode, position

def dog2_mode(landmark):
    class_mode = False

    s_angle = prc.spread_angle(landmark)
    state = prc.spread_state(s_angle)

    if state==[1,0,0,1]:
        class_mode = True
    position = [ [landmark[8].x, landmark[8].y] ]

    return class_mode, position

def dog3_mode(landmark):
    class_mode = True
    position = [ [landmark[8].x, landmark[8].y] ]

    return class_mode, position


# TWOHAND detail -------------------------------------------
def butterfly_mode(landmark1, landmark2):
    class_mode = False

    s_angle1 = prc.spread_angle(landmark1)
    s_angle2 = prc.spread_angle(landmark2)
    state1 = prc.spread_state(s_angle1)
    state2 = prc.spread_state(s_angle2)

    mid_pt = [(landmark1[0].x + landmark2[0].x)/2, (landmark1[0].y + landmark2[0].y)/2, (landmark1[0].z + landmark2[0].z)/2]
    #mid_angle = prc.calc_angle(prc.point_to_list(landmark1[9]), prc.point_to_list(landmark2[9]), mid_pt)

    #if state1==[1,1,1,1] and state2==[1,1,1,1] and mid_angle>=90:
        #class_mode = True
    if state1==[1,1,1,1] and state2==[1,1,1,1]:
        class_mode = True
    position = [mid_pt[0], mid_pt[1]] # 객체 삽입 위치

    return class_mode, position

def snail_mode(landmark1, landmark2):
    class_mode = True
    if landmark1[8].y < landmark2[8].y:
        position = [landmark1[8].x, landmark1[8].y]
    elif landmark1[8].y > landmark2[8].y:
        position = [landmark2[8].x, landmark2[8].y]
    return class_mode, position

def deer_mode(landmark1, landmark2):
    class_mode= True
    if landmark1[8].y < landmark2[8].y:
        position = [landmark1[8].x, landmark1[8].y]
    elif landmark1[8].y > landmark2[8].y:
        position = [landmark2[8].x, landmark2[8].y]

    return class_mode, position

def heart_mode(landmark1, landmark2):
    class_mode = True
    position = [(landmark1[8].x + landmark2[8].x)/2, (landmark1[8].y + landmark2[8].y)/2]
    return class_mode, position

def duck_mode(landmark1, landmark2):
    class_mode = True
    if landmark1[8].y < landmark2[8].y:
        position = [landmark1[8].x, landmark1[8].y]
    elif landmark1[8].y > landmark2[8].y:
        position = [landmark2[8].x, landmark2[8].y]
    return class_mode, position

def sun_mode(landmark1, landmark2):
    class_mode = True
    position = [(landmark1[8].x + landmark2[8].x)/2, (landmark1[8].y + landmark2[8].y)/2]
    return class_mode, position

def house_mode(landmark1, landmark2):
    class_mode = False

    s_angle1 = prc.spread_angle(landmark1)
    s_angle2 = prc.spread_angle(landmark2)
    state1 = prc.spread_state(s_angle1)
    state2 = prc.spread_state(s_angle2)

    if state1[0]==1 and state2[0]==1:
        class_mode = True
    position = [landmark1[4].x, landmark1[4].y]

    return class_mode, position

def tree_mode(landmark1, landmark2):
    class_mode = True

    if landmark1[0].y < landmark2[0].y:
        position = [landmark1[0].x, landmark1[0].y]
    elif landmark1[0].y > landmark2[0].y:
        position = [landmark2[0].x, landmark2[0].y]

    return class_mode, position

def flower_mode(landmark1, landmark2):
    #class_mode = False
    class_mode = True

    s_angle1 = prc.spread_angle(landmark1)
    s_angle2 = prc.spread_angle(landmark2)
    state1 = prc.spread_state(s_angle1)
    state2 = prc.spread_state(s_angle2)

    #if (state1==[1,0,0,1] and state2==[0,0,0,0]) or (state1==[0,0,0,0] and state2==[1,0,0,1]):
        #class_mode = True

    if landmark1[0].y < landmark2[0].y:
        position = [landmark1[0].x, landmark1[0].y]
    elif landmark1[0].y > landmark2[0].y:
        position = [landmark2[0].x, landmark2[0].y]
    
    return class_mode, position