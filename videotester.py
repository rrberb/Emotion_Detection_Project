import os
import time
import cv2
import numpy as np
import warnings
import webbrowser
warnings.filterwarnings("ignore")

from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.applications.mobilenet import preprocess_input

# load model
model = load_model(r"C:\Users\admin\Desktop\Final Project\Emotion-detection-main\best_model.h5")

# Haar cascade for face detection
face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

# 🎵 Song dictionary
songs = {
    "happy": {
        "english": ["Happy - Pharrell Williams", "Uptown Funk - Bruno Mars", "Can't Stop the Feeling - Justin Timberlake"],
        "hindi": ["Gallan Goodiyan - Dil Dhadakne Do", "London Thumakda - Queen", "Kar Gayi Chull - Kapoor & Sons"]
    },
    "sad": {
        "english": ["Someone Like You - Adele", "Fix You - Coldplay", "Let Her Go - Passenger"],
        "hindi": ["Channa Mereya - Ae Dil Hai Mushkil", "Tadap Tadap - Hum Dil De Chuke Sanam", "Agar Tum Saath Ho - Tamasha"]
    },
    "neutral": {
        "english": ["Shape of You - Ed Sheeran", "Closer - Chainsmokers", "Perfect - Ed Sheeran"],
        "hindi": ["Tum Hi Ho - Aashiqui 2", "Pee Loon - Once Upon A Time in Mumbaai", "Jeene Laga Hoon - Ramaiya Vastavaiya"]
    },
    "surprise": {
        "english": ["Shake It Off - Taylor Swift", "Thunder - Imagine Dragons", "On Top of the World - Imagine Dragons"],
        "hindi": ["Badtameez Dil - Yeh Jawaani Hai Deewani", "Zinda - Bhaag Milkha Bhaag", "Jee Karda - Singh is Kinng"]
    },
    "fear": {
        "english": ["Demons - Imagine Dragons", "In the End - Linkin Park", "Creep - Radiohead"],
        "hindi": ["Naina - Khoobsurat", "Bhula Dena - Aashiqui 2", "Yaad Hai Na - Raaz Reboot"]
    }
}

# 🟢 Choose language once (English / Hindi)
language = input("Choose your language (english/hindi): ").strip().lower()
if language not in ["english", "hindi"]:
    language = "english"

start_time = time.time()
detected_emotion = None

while True:
    ret, test_img = cap.read()
    if not ret:
        continue

    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        roi_color = test_img[y:y + h, x:x + w]
        roi_resized = cv2.resize(roi_color, (224, 224))

        img_pixels = img_to_array(roi_resized)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels = preprocess_input(img_pixels)

        predictions = model.predict(img_pixels)
        max_index = np.argmax(predictions[0])
        detected_emotion = emotions[max_index]

        confidence = predictions[0][max_index] * 100
        text = f"{detected_emotion} ({confidence:.1f}%)"
        cv2.putText(test_img, text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Facial emotion analysis', resized_img)

    # ✅ After 6 seconds, suggest songs and exit
    if time.time() - start_time >= 6 and detected_emotion is not None:
        if detected_emotion in songs:
            suggested = songs[detected_emotion][language]
            query = f"{suggested[0]} {language} song"
            url = f"https://www.youtube.com/results?search_query={query}"
            print(f"Detected Emotion: {detected_emotion}")
            print(f"Suggesting: {suggested[0]}")
            webbrowser.open_new_tab(url)

        cap.release()
        cv2.destroyAllWindows()
        break

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
