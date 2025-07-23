import cv2

#initialise person detector
hog =cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cap = cv2.VideoCapture(0)

#while loop for detections
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    #frame size
    frame = cv2.resize(frame, (640, 480))
    
    #detect people 
    (rects, weights) = hog.detectMultiScale(frame, winStride=(8, 8),
                                            padding=(8, 8), scale=1.05)
    
    #green borders
    for (x, y, w, h) in rects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0))
        
    cv2.imshow("Figure detect", frame)
        
    key = cv2.waitKey(1)
    if key == 27:
        break
    
cap.release()
cv2.destroyAllWindows()