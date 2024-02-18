from ultralytics import YOLO
import cv2
import math
import pyttsx3
def video_detection(path_x):
    video_capture = path_x
    cap=cv2.VideoCapture(video_capture)
    # text_speech = pyttsx3.init()


    model=YOLO("best.pt")

    classNames = ["Pothole"]


    while True:
        success, img = cap.read()
        # Doing detections using YOLOv8 frame by frame
        results=model(img,stream=True)

        for r in results:
            # boxes: ultralytics.engine.results.Boxes object
            boxes=r.boxes
            for box in boxes:
                #print(box)
                x1,y1,x2,y2=box.xyxy[0]
                # print(x1, y1, x2, y2)
                x1,y1,x2,y2=int(x1), int(y1), int(x2), int(y2)
                print(x1,y1,x2,y2)
                cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,255),3)
                print(box.conf[0])
                conf=math.ceil((box.conf[0]*100))/100
                print(conf)
                cls = int(box.cls[0])
                class_name = classNames[cls]
                # if class_name == 'Pothole':
                #     text_speech.say('Pothole Ahead')
                    # text_speech.runAndWait()
                    # text_speech.setProperty('rate', 100)
                label = f'{class_name}{conf}'
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=1)[0]
                # print(t_size)
                # print('c2 :- ',x1+t_size[0])
                cv2.rectangle(img, (x1, y1), (x2, y2), [255, 0, 255], 0, cv2.LINE_AA)
                cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

        yield img

cv2.destroyAllWindows()