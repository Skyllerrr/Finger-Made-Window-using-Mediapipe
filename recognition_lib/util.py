import cv2
import math

def make_dict_hand():
    
    dict_hand0 = {i:0 for i in range(16)}
    dict_hand1 = {i:0 for i in range(16)}
    dict_hand2 = {i:0 for i in range(16)}
    dict_hand3 = {i:0 for i in range(16)}
    dict_hand4 = {i:0 for i in range(16)}
    dict_hand5 = {i:0 for i in range(16)}
    dict_hand = [dict_hand0, dict_hand1, dict_hand2, dict_hand3, dict_hand4, dict_hand5]

    return dict_hand

def input_name(label, image):

    if label==0:
        cv2.putText(image, "Butterfly", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)

    elif label==1:
        cv2.putText(image, "cat", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)

    elif label==2:
        cv2.putText(image, "snail", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)

    elif label==3:
        cv2.putText(image, "deer", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)

    elif label==4:
        cv2.putText(image, "heart", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)

    elif label==5:
        cv2.putText(image, "duck", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)

    elif label==6:
        cv2.putText(image, "sun", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)

    elif label==7:
        cv2.putText(image, "house", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)

    elif label==8:
        cv2.putText(image, "tree", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)

    elif label==9:
        cv2.putText(image, "rock", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)

    elif label==10:
        cv2.putText(image, "flower", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)

    elif label==11:
        cv2.putText(image, "Dog1", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)

    elif label==12:
        cv2.putText(image, "Dog2", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)

    elif label==13:
        cv2.putText(image, "Dog3", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)

    elif label==15:
        cv2.putText(image, "STOP!!!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)

    else:
        cv2.putText(image, "Make a shape!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)