import math
import numpy as np

def point_to_list(landmark_pt): #  (.x .y .z) -> [x, y, z]
    # 추가 기능들의 계산, 사용편의를 위해 mediapipe 에서 landmark(.x .y .z)의 데이터를 list[x, y, z]로 변환
    pt_list = [landmark_pt.x, landmark_pt.y, landmark_pt.z]
    return pt_list


def get_distance(point1, point2): # 두 점 사이의 거리 반환

    pt1 = list(map(lambda n: n*100, point1))
    pt1 = list(map(int, pt1))
    pt2 = list(map(lambda n: n*100, point2))
    pt2 = list(map(int, pt2))

    distance = math.sqrt((pt2[0]-pt1[0])**2 + (pt2[1]-pt1[1])**2 + (pt2[2]-pt1[2])**2)

    return distance # 100 배율 값


def calc_angle(pt1, pt2, pt3): # 두 벡터가 이루는 각(degree값)을 세점으로 부터 구해 반환
    # pt3가 두 벡터의 공통 좌표
    # .[0]: x좌표  |  .[1]: y좌표  |  .[2]: z좌표
    # theta 구한 후 degree로 변환

    theta = np.arccos(
            (((pt1[0]-pt3[0])*(pt2[0]-pt3[0])) +
            ((pt1[1]-pt3[1])*(pt2[1]-pt3[1])) +
            ((pt1[2]-pt3[2])*(pt2[2]-pt3[2]))) /
            (math.sqrt((pt1[0]-pt3[0])**2 + (pt1[1]-pt3[1])**2 + (pt1[2]-pt3[2])**2) *
            math.sqrt((pt2[0]-pt3[0])**2 + (pt2[1]-pt3[1])**2 + (pt2[2]-pt3[2])**2))
            )

    degree = math.degrees(theta)

    return degree


def spread_angle(landmark): # 손가락 사이의 펼침 각도 반환

    # 손가락 사이 각도 계산을 위한 중간점(좌표)
    middle_pt1 = [(landmark[2].x + landmark[5].x)/2, (landmark[2].y + landmark[5].y)/2, (landmark[2].z + landmark[5].z)/2]
    middle_pt2 = [(landmark[5].x + landmark[9].x)/2, (landmark[5].y + landmark[9].y)/2, (landmark[5].z + landmark[9].z)/2]
    middle_pt3 = [(landmark[9].x + landmark[13].x)/2, (landmark[9].y + landmark[13].y)/2, (landmark[9].z + landmark[13].z)/2]
    middle_pt4 = [(landmark[13].x + landmark[17].x)/2, (landmark[13].y + landmark[17].y)/2, (landmark[13].z + landmark[17].z)/2]

    angle1 = calc_angle(point_to_list(landmark[3]), point_to_list(landmark[5]), middle_pt1)
    angle2 = calc_angle(point_to_list(landmark[6]), point_to_list(landmark[10]), middle_pt2)
    angle3 = calc_angle(point_to_list(landmark[10]), point_to_list(landmark[14]), middle_pt3)
    angle4 = calc_angle(point_to_list(landmark[14]), point_to_list(landmark[18]), middle_pt4)
    angle = [angle1, angle2, angle3, angle4]

    return angle


def spread_state(angle): # 손가락 펼침 상태 반환

    # 손가락 사이의 각도에 따른 상태판단으로 벌리면1 아니면0
    angle_state = [0,0,0,0]

    # 벌리는 기준 각도 설정 - 만족시에 1의값
    if (angle[0] > 30):
        angle_state[0] = 1
    if (angle[1] > 20):
        angle_state[1] = 1
    if (angle[2] > 20):
        angle_state[2] = 1
    if (angle[3] > 20):
        angle_state[3] = 1

    return angle_state


def fold_angle(landmark): # 손가락 관절의 굽힘 각도 반환

    landmark_p = []
    for i in landmark:
        landmark_p.append(point_to_list(i))
        
    f_angle0 = calc_angle(landmark_p[0], landmark_p[2], landmark_p[1]) #landmark[1] 위치 각도
    f_angle1 = calc_angle(landmark_p[1], landmark_p[3], landmark_p[2]) #landmark[2] 위치 각도
    f_angle2 = calc_angle(landmark_p[2], landmark_p[4], landmark_p[3]) #landmark[3] 위치 각도
    f_angle3 = calc_angle(landmark_p[0], landmark_p[6], landmark_p[5]) #landmark[5] 위치 각도
    f_angle4 = calc_angle(landmark_p[5], landmark_p[7], landmark_p[6]) # ...
    f_angle5 = calc_angle(landmark_p[6], landmark_p[8], landmark_p[7])
    f_angle6 = calc_angle(landmark_p[0], landmark_p[10], landmark_p[9])
    f_angle7 = calc_angle(landmark_p[9], landmark_p[11], landmark_p[10])
    f_angle8 = calc_angle(landmark_p[10], landmark_p[12], landmark_p[11])
    f_angle9 = calc_angle(landmark_p[0], landmark_p[14], landmark_p[13])
    f_angle10 = calc_angle(landmark_p[13], landmark_p[15], landmark_p[14])
    f_angle11 = calc_angle(landmark_p[14], landmark_p[16], landmark_p[15])
    f_angle12 = calc_angle(landmark_p[0], landmark_p[18], landmark_p[17])
    f_angle13 = calc_angle(landmark_p[17], landmark_p[19], landmark_p[18])
    f_angle14 = calc_angle(landmark_p[18], landmark_p[20], landmark_p[19])

    f_angle = [f_angle0, f_angle1, f_angle2, f_angle3, f_angle4, f_angle5, f_angle6, f_angle7,
                f_angle8, f_angle9, f_angle10, f_angle11, f_angle12, f_angle13, f_angle14]

    return f_angle


def fold_state(angle): # 손가락 굽힘 상태 반환

    # 손가락 구부리는 각도에따라 완전 접었는지 폈는지 판단
    angle_state = [0,0,0,0,0]

    if (angle[1] < 170) and (angle[2] < 125): # 엄지
        angle_state[0] = 1
    if (angle[3] < 150) and (angle[4] < 90) and (angle[5] < 150): # 검지
        angle_state[1] = 1
    if (angle[6] < 150) and (angle[7] < 90) and (angle[8] < 150): # 중지
        angle_state[2] = 1
    if (angle[9] < 150) and (angle[10] < 90) and (angle[11] < 150): # 약지
        angle_state[3] = 1
    if (angle[12] < 150) and (angle[13] < 90) and (angle[14] < 150): # 소지
        angle_state[4] = 1

    return angle_state