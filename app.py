import numpy as np
import cv2
from keras.models import load_model


model = load_model('leon.h5')

def keras_predict(model, image):
    processed = keras_process_image(image)
    steering_angle = float(model.predict(processed, batch_size=1))
    steering_angle = steering_angle * 60
    return steering_angle


def keras_process_image(img):
    image_x = 100
    image_y = 100
    img = cv2.resize(img, (image_x, image_y))
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, (-1, image_x, image_y, 1))
    return img


steer = cv2.imread('steering_wheel_image.jpg', 0)
rows, cols = steer.shape
smoothed_angle = 0

cap = cv2.VideoCapture('pommes.mp4')
while (cv2.waitKey(10) != ord('q')):
    ret, frame = cap.read()
    gray = cv2.resize(frame[:, :, 1], (320, 180))
    steering_angle = keras_predict(model, gray)
    low_yellow = np.array([18, 94, 160])
    up_yellow = np.array([255, 255, 255])

    mask = cv2.inRange(frame, low_yellow, up_yellow)
    edges = cv2.Canny(mask, 75, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, maxLineGap=2)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (255, 105, 180), 3)

    print(steering_angle)
    cv2.imshow('frame', cv2.resize(frame, (600, 400), interpolation=cv2.INTER_AREA))
    smoothed_angle += 0.2 * pow(abs((steering_angle - smoothed_angle)), 2.0 / 3.0) * (
        steering_angle - smoothed_angle) / abs(
        steering_angle - smoothed_angle)
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -smoothed_angle, 1)
    dst = cv2.warpAffine(steer, M, (cols, rows))
    cv2.imshow("steering wheel", dst)
    cv2.imshow("edges", edges)
    #img_all = np.concatenate((frame,dst,edges),axis=0)
    #cv2.imshow("Pilot viewer",img_all)
cap.release()
cv2.destroyAllWindows()
