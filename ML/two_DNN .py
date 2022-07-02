import cv2
import mediapipe as mp
import numpy as np
import keras

from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

gesture = {
    0:'butterfly', 1:'cat', 2:'snail', 3:'deer', 4:'heart', 5:'elephant',
    6:'sun', 7:'house', 8:'tree', 9:'rock', 10:'flower', 11:'dog1', 12:'dog2', 13:'dog3'
}
'''
gesture = {
    0=0:'Butterfly', 2=1:'snail', 3=2:'deer', 4=3:'heart', 5=4:'elephant'
    6=5:'sun', 7=6:'house', 8=7:'tree', 10=8:'flower'
}
'''

file = np.genfromtxt('C:/Users/SMAI-JH/Documents/Coding/capstone03/ML3/dataset_twohand_duck_flower.csv', delimiter=',')
angle = file[:,:-1].astype(np.float32)
label = file[:, -1].astype(np.float32)

angle_ar = np.array(angle)
label_ar = np.array(label)

print(angle_ar.shape)
print(label_ar.shape)

x_train, x_val, y_train, y_val = train_test_split(angle_ar, label_ar, random_state=42, test_size=0.2)

print(x_train.shape)
print(y_train.shape)
print(x_val.shape)
print(y_val.shape)

y_train = np_utils.to_categorical(y_train)
y_val = np_utils.to_categorical(y_val)

# 모델 생성
from keras import layers
model = keras.models.Sequential()
model.add(Dense(32, input_shape=(30,), activation='relu'))
model.add(layers.Dropout(0.5))
model.add(Dense(16, input_shape=(30,), activation='relu'))
model.add(Dense(9, activation="softmax"))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  

hist = model.fit(x_train, y_train, epochs=30, batch_size=16, validation_data=(x_val, y_val))
model.save('C:/Users/SMAI-JH/Documents/Coding/capstone03/ML3/two_DNN_duck_flower2.h5')

# 5. 학습과정 살펴보기
print('## training loss and acc ##')
print(hist.history['loss'])
print(hist.history['accuracy'])

# 6. 모델 평가하기
loss_and_metrics = model.evaluate(x_val, y_val)
print('## evaluation loss and_metrics ##')
print(loss_and_metrics)

# 7. 모델 그래프
acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

'''
# 7. 모델 그래프 - SMOOTH
from scipy.interpolate import interp1d

acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

epochs = range(1, len(acc) + 1)

cubic_interploation_model_acc=interp1d(epochs,acc,kind="cubic")
accs = np.linspace(1,epochs,100)
y_accs=cubic_interploation_model_acc(accs)

cubic_interploation_model_val_acc=interp1d(epochs,val_acc,kind="cubic")
val_accs = np.linspace(1,epochs,100)
y_val_accs=cubic_interploation_model_val_acc(val_accs)

cubic_interploation_model_loss=interp1d(epochs,loss,kind="cubic")
losss = np.linspace(1,epochs,100)
y_losss=cubic_interploation_model_loss(losss)

cubic_interploation_model_val_loss=interp1d(epochs,val_loss,kind="cubic")
val_losss = np.linspace(1,epochs,100)
y_val_losss=cubic_interploation_model_val_loss(val_losss)



plt.plot(accs, y_accs, 'r', label='Training acc')
plt.plot(val_accs, y_val_accs, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(losss, y_losss, 'r', label='Training loss')
plt.plot(val_losss, y_val_losss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
'''