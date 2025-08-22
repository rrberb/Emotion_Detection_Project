import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all logs, 1 = filter INFO, 2 = filter WARNING, 3 = filter ERROR
import cv2
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.applications.mobilenet import preprocess_input   # ✅ MobileNet preprocessing

# load model
model = load_model(r"C:\Users\admin\Desktop\Final Project\Emotion-detection-main\best_model.h5")

# Haar cascade for face detection
face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

while True:
    ret, test_img = cap.read()
    if not ret:
        continue

    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # crop ROI in color (MobileNet needs RGB, 224x224)
        roi_color = test_img[y:y + h, x:x + w]
        roi_resized = cv2.resize(roi_color, (224, 224))

        # preprocess for MobileNet
        img_pixels = img_to_array(roi_resized)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels = preprocess_input(img_pixels)   # ✅ scales to [-1, 1]

        # predict emotion
        predictions = model.predict(img_pixels)
        max_index = np.argmax(predictions[0])
        predicted_emotion = emotions[max_index]

        # get confidence (probability of predicted class)
        confidence = predictions[0][max_index] * 100

        # display text: emotion + confidence
        text = f"{predicted_emotion} ({confidence:.1f}%)"
        cv2.putText(test_img, text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Facial emotion analysis', resized_img)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
