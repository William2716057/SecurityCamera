import cv2
import pyaudio
import numpy as np

#tone for detection
def playTone():
    p = pyaudio.PyAudio()
    volume = 0.5     # range [0.0, 1.0]
    fs = 44100       # sampling rate, Hz
    duration = 0.2   # seconds
    f = 440.0        # sound frequency (Hz)

    # generate sine wave tone
    samples = (np.sin(2*np.pi*np.arange(fs*duration)*f/fs)).astype(np.float32)

    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=fs,
                    output=True)

    stream.write(volume*samples)
    stream.stop_stream()
    stream.close()
    p.terminate()

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
        
    #if detected = True:
    detected = False
    for (x, y, w, h) in rects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        detected = True

    # If any person detected, play tone
    if detected:
        playTone()

    cv2.imshow("Figure detect", frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()