import cv2 as cv
from variables import*

def object_detection(CascadeClassifier, img):
    classifier = CascadeClassifier(haarcascade_path)
    bboxes = classifier.detectMultiScale(img)
    for box in bboxes:
        x, y, width, height = box
        x2, y2 = x + width, y + height
        cv.rectangle(img, (x, y), (x2, y2), (0,0,255), 1)
    return img