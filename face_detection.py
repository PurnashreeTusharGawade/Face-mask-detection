import numpy as np
import cv2

capture = cv2.VideoCapture(0)
data_without_mask = []
data_with_mask = []
haar_data = cv2.CascadeClassifier('data.xml')

while True:
    flag, img = capture.read()
    if flag:
        faces = haar_data.detectMultiScale(img)
        for x, y, w, h in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 4)
            face = img[y:y + h, x:x + w, :]
            face = cv2.resize(face, (50, 50))
            print("Total samples:", len(data_without_mask) + len(data_with_mask))
            if len(data_without_mask) < 200:
                data_without_mask.append(face)
            elif len(data_with_mask) < 200:
                data_with_mask.append(face)
        cv2.imshow('result', img)
        if cv2.waitKey(2) == 27 or (len(data_without_mask) >= 200 and len(data_with_mask) >= 200):
            break

capture.release()
cv2.destroyAllWindows()

np.save("without_mask.npy", data_without_mask)
np.save("with_mask.npy", data_with_mask)

