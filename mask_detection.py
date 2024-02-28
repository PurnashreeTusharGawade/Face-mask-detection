import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Load the saved numpy arrays
with_mask = np.load('with_mask.npy')
without_mask = np.load('without_mask.npy')

# Flatten the images
with_mask = with_mask.reshape(200, 50 * 50 * 3)
without_mask = without_mask.reshape(200, 50 * 50 * 3)

# Concatenate the flattened arrays
X = np.r_[with_mask, without_mask]

# Create labels
labels = np.zeros(X.shape[0])
labels[200:] = 1.0

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, labels, test_size=0.25)

# Train SVM model
svm = SVC()
svm.fit(x_train, y_train)

# Load Haar cascade classifier
haar_data = cv2.CascadeClassifier('data.xml')

# Function to predict and draw rectangles
def predict_and_draw(frame):
    faces = haar_data.detectMultiScale(frame)
    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 4)
        face = cv2.resize(frame[y:y+h, x:x+w], (50, 50)).reshape(1, -1)
        pred = svm.predict(face)
        label = "Mask" if pred[0] == 0 else "No Mask"  # Accessing the first element of pred
        cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (244, 250, 250), 2)
    return frame

# Video capture
capture = cv2.VideoCapture(0)
while True:
    flag, img = capture.read()
    if flag:
        img = predict_and_draw(img)
        cv2.imshow('result', img)
        if cv2.waitKey(2) == 27:
            break

capture.release()
cv2.destroyAllWindows()

