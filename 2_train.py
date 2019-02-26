#Run 1_read.py in interactive mode before running this script

from keras.models import Sequential
from keras.layers import Dense, Activation
import matplotlib.pyplot as plt

model = Sequential()
model.add(Dense(32, input_dim=784))
model.add(Activation('relu'))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('sigmoid'))
model.compile(optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

history = model.fit(train_arr, train_labels,batch_size=128,epochs=64,validation_data=(test_arr,test_labels), verbose=2)
#epochs = number of times it used the whole set of sample
#batch_size = number of samples used for each fit; the smaller it is, the better. But if too small -> too random; too big -> trapped in local minimum.
model.save("mnist_model.h5")

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training loss', 'validation loss'])
plt.savefig("loss.png")
plt.close()

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['training accuracy', 'validation accuracy'])
plt.savefig("accuracy.png")
plt.close()


