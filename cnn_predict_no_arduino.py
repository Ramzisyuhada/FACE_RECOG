# --------- load Haar Cascade model -------------
import cv2
from keras.src.saving.saving_api import load_model
import serial
import time

face_cascade = cv2.CascadeClassifier('../../Cascade/haarcascade_frontalface_default.xml')


# serialArduino = serial.Serial('COM9', 9600)
# --------- load Keras CNN model -------------
model = load_model("model_terlatih.h5")
print("[INFO] finish load model...")

def draw_ped(img, label, x0, y0, xt, yt, color=(255,127,0), text_color=(255,255,255)):

    (w, h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img,
                  (x0, y0 + baseline),
                  (max(xt, x0 + w), yt),
                  color,
                  2)
    cv2.rectangle(img,
                  (x0, y0 - h),
                  (x0 + w, y0 + baseline),
                  color,
                  -1)
    cv2.putText(img,
                label,
                (x0, y0),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                text_color,
                1,
                cv2.LINE_AA)
    return img
labels = ["Ramzi", "Tidak"]
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        for (x, y, w, h) in faces:

            face_img = gray[y:y + h, x:x + w]
            face_img = cv2.resize(face_img, (50, 50),interpolation=cv2.INTER_LINEAR)
            face_img = face_img.reshape(1, 50, 50, 1)

            result = model.predict(face_img)
            idx = int(result.argmax(axis=1))
            confidence = result.max(axis=1) * 100
            if confidence > 80:
                label_text = "%s (%.2f %%)" % (labels[idx], confidence)
                # if labels[idx] == "Ramzi":
                #     serialArduino.write(b'0')
                # else:
                #     serialArduino.write(b'1')
            else:
                label_text = "N/A"


            frame = draw_ped(frame, label_text, x, y, x + w, y + h, color=(0, 255, 255), text_color=(50, 50, 50))

        cv2.imshow('Detect Face', frame)
    else:
        break
    if cv2.waitKey(10) == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()