from ultralytics import YOLO
import cv2
import math 

# start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# model
model = YOLO("D:/wepons/content/ultralytics/runs/detect/train/weights/best.pt")

# object classes
classNames = ["SUSPICIOUS BJECT DETECTED-STAY ALERT", "SUSPICIOUS BJECT DETECTED-STAY ALERT",
              "SUSPICIOUS BJECT DETECTED-STAY ALERT", "SUSPICIOUS BJECT DETECTED-STAY ALERT",
              "SUSPICIOUS BJECT DETECTED-STAY ALERT", "SUSPICIOUS BJECT DETECTED-STAY ALERT",
              "SUSPICIOUS BJECT DETECTED-STAY ALERT", "SUSPICIOUS BJECT DETECTED-STAY ALERT",
              "SUSPICIOUS BJECT DETECTED-STAY ALERT"]


while True:
    success, img = cap.read()
    results = model(img, stream=True, confidence=0.65)

    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # put box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            print("Accuracy--->", confidence)

            # class name
            cls = int(box.cls[0])
            print("Class name -->", classNames[cls])

            # object details
            text = classNames[cls] + " - Accuracy: {:.2f}".format(confidence)
            org = (x1, y1 - 10)
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.5
            color = (255, 0, 0)
            thickness = 1

            # calculate text size
            (text_width, text_height), baseline = cv2.getTextSize(text, font, fontScale, thickness)

            # draw red box
            cv2.rectangle(img, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 0, 255), -1)

            # put text
            cv2.putText(img, text, org, font, fontScale, color, thickness)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
