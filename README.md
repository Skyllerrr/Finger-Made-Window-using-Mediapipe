# Finger Made Window Using Mediapipe

## **Mediapipe**를 이용한 유아용 놀이 교구

<br>

영유아기에 미디어를 접하는 정도는 상당 수를 확인할 수 있는 와중에 미디어를 활용한 교육방식은 현대사회 부모들에게 각광받는 교육수단임에 틀림없다.

따라서, 프로젝트 진행은 **Python기반 OpenCV 라이브러리**를 사용하여 디스플레이상에 영상을 송출하는 바, <br>화면 송출 알고리즘을 구성하여 만든 프로젝트이다.

<br>
<br>

![image](https://github.com/Skyllerrr/Finger-Made-Window-using-Mediapipe/assets/93968241/4a5085da-ac64-4002-8bdc-5d2ba95d047e)

### **[Hands MediaPipe]**

위 사진은 **사람의 손을 MediaPipe로** 나타낸 사진이다. 사용자가 웹캠에 띄워지는 MediaPipe를 이용하여 임의의 정적 혹은 동적인 객체를 손 동작으로 표현하게 되면, 손 동작에 맞는 관절점과 각도에 맞게 해당하는 객체를 인식하여 송출되는 화면에 객체가 찍히는 과정으로 이루어져 있다.

프로젝트에서는 2가지의 파트로 나누어져있다. 사용자의 손 동작에 따라 정적 및 동적 객체를 관절점과 각도에 맞게 표현하면 임의의 좌표가 출력이 되는 **인식부**와 사용자의 손 동작에 따른 객체를 인식하여 출력된 좌표값으로 화면에 해당하는 객체를 띄워주는 **송출부**로 나뉜다.

<br>
<br>

## 인식부

![image](https://github.com/Skyllerrr/Finger-Made-Window-using-Mediapipe/assets/93968241/7723c739-dff6-44c0-8682-4b7a966d944b)

**<사용자가 각각의 조건을 따라 동적인 객체인 동물(강아지)을 손으로 표현하는 과정의 예시 사진>**

* **조건 1** : 엄지와 검지, 검지와 중지가 서로 벌려져있고 나머지 손가락이 붙어있을 때의 조건
* **조건 2** : 약지, 중지, 엄지를 모았을 때의 조건
* **조건 3** : 엄지와 검지, 약지와 새끼가 서로 벌려져있고 검지 손가락 끝 관절이 일정 각도보다 작을 때의 조건

<br>
<br>

## 송출부

![나비 객체 송출](https://github.com/Skyllerrr/Finger-Made-Window-using-Mediapipe/assets/93968241/b9a5386b-b92b-48db-a035-71a587afb1e8)

**<사용자가 직접 손으로 동적인 객체(나비)를 표현하여 화면에 찍히는 과정의 예시 사진>**


