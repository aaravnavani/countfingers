import cv2
import numpy as np
import math
import time



 
learningRate = 0

isBgCaptured = False  
def removeBG(frame):
    fgmask = bgModel.apply(frame,learningRate=learningRate)

    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res

def calculateFingers(result, drawing):
    hull = cv2.convexHull(result, returnPoints = False)
    #convexityDefects function gets all the defects of the contour and we save them in another array 
    #Defects are the lowest points between one finger and the other 
    #convexityDefects returns more points than we need, so we have to filter them. We filter them based on their distance from the center of the rectangle.
    #Therefore, only the lowest points between each finger are kept 
    if len(hull) > 3:
        defects = cv2.convexityDefects(result, hull)
        if type(defects) != type(None):  # avoid crashing.   (BUG not found)
            count = 0
            for i in range(defects.shape[0]):
                s,e,f,d = defects[i,0]

                start = tuple(result[s][0])
                end = tuple(result[e][0])
                far = tuple(result[f][0])

                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)

                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                
                #Law of Cosines 
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))
                if angle <= math.pi / 2:  
                    count += 1
                    cv2.circle(drawing, far, 8, [255, 255, 255], -1)
            return True, count
    return False, 0

camera = cv2.VideoCapture(0)

while camera.isOpened():
    ret, frame = camera.read()
    frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothen image
    frame = cv2.flip(frame, 1)  # flip the frame horizontally
    cv2.rectangle(frame, (int(0.5 * frame.shape[1]), 0),
                 (frame.shape[1], int(0.8 * frame.shape[0])), (255, 0, 0), 2)
    cv2.imshow('original', frame)

    if isBgCaptured == True:  
        img = removeBG(frame)
        img = img[0:int(0.8 * frame.shape[0]),
                    int(0.5 * frame.shape[1]):frame.shape[1]]  # clip the ROI

        # convert the image into binary image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (41, 41), 0)
        ret, thresh = cv2.threshold(blur , 60, 255, cv2.THRESH_BINARY)


        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        maxArea = -1
        if len(contours) > 0: 
            for i in range(len(contours)): 
                temp = contours[i]
                area = cv2.contourArea(temp)
                if area > maxArea: 
                    maxArea = area 
                    index = i

            result = contours[index]
            hull = cv2.convexHull(result)
            drawing = np.zeros(img.shape, np.uint8)
            cv2.drawContours(drawing, [result], 0, (0, 255, 0), 2)
            cv2.drawContours(drawing, [hull], 0, (0, 255, 0), 2)

            isFinishCal,count = calculateFingers(result,drawing)
            

        cv2.putText(drawing, str(count+1), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
        cv2.imshow('drawing', drawing)

       
    k = cv2.waitKey(10)
    if k == 27:  # press ESC to exit
        camera.release()
        cv2.destroyAllWindows()
        break
    elif k == ord('b'):  # press 'b' to capture the background
        bgModel = cv2.createBackgroundSubtractorMOG2(0, 50)
        isBgCaptured = True