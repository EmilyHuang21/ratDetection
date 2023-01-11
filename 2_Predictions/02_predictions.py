import cv2
from yolo_prediction import YOLO_Pred
yolo = YOLO_Pred('D:/ratDetection/2_Predictions/Model4/weights/best.onnx',
                 'D:/ratDetection/2_Predictions/data.yml')

img = cv2.imread('D:/ratDetection/test_pic1.jpg')

'''cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()'''

# predictions
img_pred = yolo.predictions(img)
cv2.imshow('prediction image', img_pred)
cv2.waitKey(0)
cv2.destroyAllWindows()

# video
cap = cv2.VideoCapture('D:/ratDetection/test_video1.mp4')
while True:
    ret, frame = cap.read()
    if ret == False:
        print('unable to read video')
        break

    pred_image = yolo.predictions(frame)

    cv2.imshow('YOLO', pred_image)
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
cap.release()

webcam_video_stream = cv2.VideoCapture(1)
while True:
    ret, frame = webcam_video_stream.read()
    if ret == False:
        print('unable to read video')
        break

    pred_image = yolo.predictions(frame)

    cv2.imshow('YOLO', pred_image)
    if cv2.waitKey(1) == 27:
        break
# releasing the stream and the camera
# close all opencv windows
webcam_video_stream.release()
cv2.destroyAllWindows()
