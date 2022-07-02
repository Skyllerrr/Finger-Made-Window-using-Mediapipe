import mediapipe as mp
from recognition_lib import MLmodel
from recognition_lib import util

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

class recognition:
    
    def __init__(self, image, result, dict, counting):
        self.image = image
        self.result = result
        self.dict = dict
        self.counting = counting

        self.hand_list = []
        self.about_twohand_list = []
        self.model_list = []

        self.num = 0

        self.label = 14 # NORMAL
        self.position = [0, 0]
        self.real = []

        
    def recog_main(self):
        if len(self.hand_list)>=1 and (self.counting%10 == 0):               

            for i in range(len(self.hand_list)):
                for j in range(i+1, len(self.hand_list)):
                    distance, twohandmode = MLmodel.twohand_mode(self.hand_list[i], self.hand_list[j])

                    if twohandmode == True:
                            self.about_twohand_list.append([distance, i, j])                            

            self.about_twohand_list.sort()
            temp_list = []
            for i in self.about_twohand_list:
                if (i[1] not in temp_list) and (i[2] not in temp_list):
                    self.model_list.append( [i[1], i[2]] )
                    temp_list.append(i[1])
                    temp_list.append(i[2])      

            if len(temp_list)!=self.num+1:
                for i in range(self.num+1):
                    if i not in temp_list:
                        self.model_list.append([i])

            self.real = []
            real_label = self.predict_shape()

        return real_label


    def draw_load_hand(self):
        for self.num, hand in enumerate(self.result):
            mp_drawing.draw_landmarks(self.image, hand, mp_hands.HAND_CONNECTIONS) # 손가락 그리기
            self.hand_list.insert(0, hand.landmark) # hand list 만들기


    def predict_shape(self):
        for i in range(len(self.model_list)):
            if len(self.model_list[i]) == 2:
                #print('two hand model on!!')
                self.label, self.position = MLmodel.twohand_model(self.hand_list[self.model_list[i][0]], self.hand_list[self.model_list[i][1]])
            elif len(self.model_list[i]) == 1:
                #print('one hand model on!!')
                self.label, self.position = MLmodel.onehand_model(self.hand_list[self.model_list[i][0]])

            self.dict[i][self.label] += 1
           #print('dict[%d]' %i)
            #print(self.dict[i])
            if max(self.dict[i].values())==3:
                max_key = max(self.dict[i], key = self.dict[i].get)
                util.input_name(max_key, self.image)
                print('predict!!! dict[%d] label= %d' %(i, max_key))
                self.dict[i]= {i:0 for i in range(16)}

                self.real.append([max_key, self.position])

        return self.real