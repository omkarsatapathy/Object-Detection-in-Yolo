import cv2
import pandas as pd
import time
from ultralytics import YOLO
cap = cv2.VideoCapture(1)  # 0 for FacetimeHD camera, 1 for iPhone camera
import numpy as np
model = YOLO('yolov8n.pt', task = 'segment' )
name = model.names
time_list = [0,0,0,0]
start_time = time.time()

i = 0
while True:
    this_frame_time = time.time()

    ret, frame = cap.read()
    if i%5 == 0 or i == 0:
        results = model.predict(frame, device = 'mps')
    result = results[0]
    # print(result)
    a = pd.DataFrame(results[0].boxes.data.detach().cpu().numpy())
    # print(a)
    bboxes = np.array(result.boxes.xyxy.cpu(), dtype='int')
    classes = np.array(result.boxes.cls.cpu(), dtype='int')

    for cls, bbox, a_ in zip(classes, bboxes, a[4]):
        a_ = round((a_*100)/10, 0)*10
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (250, 250, 256))
        cv2.putText(frame, name[int(cls)]+ "  " +str(a_) + '%' ,(x1, y1 - 5), cv2.FONT_HERSHEY_PLAIN, 2, (250, 250, 250), 2)

    elapsed_time = time.time() - start_time
    time_list.append(elapsed_time)
    if i % 20 == 0 or i == 0:
        fps = round(1/ round(time_list[-1] - time_list[-2],3),0)
    cv2.putText(frame, ('Frame rate :  ') + str(fps), (60, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,0,250), 2,cv2.LINE_AA)
    time_list = time_list[-3:]
    cv2.imshow("Img", frame)

    i +=1

    if cv2.waitKey(30) == 27:
        break
