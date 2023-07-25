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

**<사용자가 직접 손으로 동적인 객체(나비)를 표현하여 화면에 찍히는 과정의 예시 영상>**

* 사용자가 **나비 모양 MediaPipe**에 맞춰 손 모양을 나타내면, 웹캠에서 손 모양을 인식하여 동작에 맞는 나비 객체를 화면에 송출해준다.

<br>
<br>

## 결과

아이들의 소근육 발달을 향상시키기 위해 아이들이 웹캠에 손 동작을 표현할 때 여러가지 손 동작을 표현하게 되면서 손 동작에 따른 감각적인 활동을 통해 송출된 객체가 무엇인지 인지하는 인지 능력이 발달되고, 손가락을 자유롭게 계속해서 움직이게 되면서 **손에 해당되는 소근육 운동 능력이 향상되는 기대효과**를 얻을 수 있다.

아이들이 웹캠에 여러가지 해당하는 객체들을 손 동작으로 표현할 때 현재 표현할 수 있는 객체의 수가 **총 12가지**로 한정되어 있는데, 한정된 객체로 인해 아이들이 표현할 수 있는 손 모양이 한정되어 있기 때문에 더 많은 동적 및 정적 객체들을 표현할 수 있는 경험이 현저히 적어진다. 아이들의 소근육 발달과 창의성 발달을 위해서는 현재 나타낼 수 있는 객체들의 수 이상으로 더 많은 객체들을 삽입하여 만들 수 있게 되면, 그만큼 아이들의 소근육 발달과 창의성 발달의 효과도 많이 높아질 것이라고 본다.

<br>

* 송출부의 객체가 송출하는 과정에서 나비를 포함하여 총 4가지의 동적 객체에 각각 손 동작을 취하면 1마리가 아닌 찍히는 프레임에 따라 <br> 2마리 이상이 송출되는 현상 **(현재 => 정상적으로 1마리씩 송출)**
* MediaPipe 인식의 불안정 **(현재 => MediaPipe 알고리즘의 리팩토링을 통해 안정화)**

