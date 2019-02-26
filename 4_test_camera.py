import numpy as np
import cv2
from keras.models import load_model

model = load_model('mnist_model.h5')
cap = cv2.VideoCapture(0)

while(True):
	# Capture frame-by-frame
	ret, frame = cap.read()
	# Operations on the frame
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	cv2.rectangle(gray,(264,184),(376,296),(0,255,0),1)
	graycrop = cv2.resize(gray[184:296,264:376],(28,28))
	graycrop = graycrop.max() - graycrop  # Color inversion
	graycrop[graycrop < 0.3 * graycrop.max()] = 0 # Image Background Subtraction
	test_arr = graycrop.flatten()
	test_arr = np.expand_dims(test_arr, axis=0)
	mo = model.predict(test_arr)
	cv2.putText(graycrop, str(np.argmax(mo[0])), (0, 8), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255))
	stringout = ''
	for i in range(10):
		stringout = stringout + str(i) + ':%2.f' %(mo[0,i]*100)+'%  '
	cv2.putText(gray, stringout, (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
	cv2.imshow('Camera',gray)
	cv2.imshow('CropForProcess',graycrop)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
# Release the capture
cap.release()
cv2.destroyAllWindows()