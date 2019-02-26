a = 25 #check from graph number a to a+9

import csv
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt

model = load_model('mnist_model.h5')

mo = model.predict(test_arr)
fig=plt.figure()
for i in range(10):
    img = np.reshape(test_arr[i+a],(28,28))
    fig.add_subplot(2, 5, i+1)
    plt.imshow(img)
    plt.title('predict: '+str(np.argmax(mo[i+a]))+'\n true: '+str(test_labels[i+a]))
plt.show()
