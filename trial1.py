# import numpy as np
import cv2

img = cv2.imread('faces4.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

face1, face2 = faces[:2]

x1, y1, w1, h1 = face1[:]
x2, y2, w2, h2 = face2[:]

roi1 = img[y1:y1+h1, x1:x1+w1]
roi2 = img[y2:y2+h2, x2:x2+w2]

swap1 = cv2.resize(roi2, (w1, h1), interpolation=cv2.INTER_CUBIC)
swap2 = cv2.resize(roi1, (w2, h2), interpolation=cv2.INTER_CUBIC)

img[y1:y1+h1, x1:x1+w1] = swap1
img[y2:y2+h2, x2:x2+w2] = swap2

cv2.imshow('swapped', img)
cv2.waitKey(0) & 0xFF
cv2.destroyAllWindows()