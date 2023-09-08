import cv2
import time
import numpy as np
import yolov5
import torch

model = yolov5.load("drone.pt")
cap = cv2.VideoCapture("test_video.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = model(img)
    predictions = result.pred[0]
    boxes = predictions[:, :4]
    x1 = int(boxes[:,0])
    y1 = int(boxes[:,1])
    x2 = int(boxes[:,2])
    y2 = int(boxes[:,3])
    print(x1,x2,y1,y2)
    scores = predictions[:,4]
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Frame", img)
    if cv2.waitKey(20) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
