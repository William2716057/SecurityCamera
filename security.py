import cv2
import pyaudio
import numpy as np
import time

# Take screenshot every 3 seconds if a person is detected
def capture_photo(frame, number):
    filename = f"photo{number}.jpg"
    cv2.imwrite(filename, frame)
    print(f"Photo saved as {filename}")

# Tone for detection
def playTone():
    p = pyaudio.PyAudio()
    volume = 0.5    
    fs = 44100       # Sampling rate, Hz
    duration = 0.5   # Shorter duration to prevent delay
    f = 900.0        # Sound frequency (Hz)

    samples = (np.sin(2 * np.pi * np.arange(fs * duration) * f / fs)).astype(np.float32)

    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=fs,
                    output=True)

    stream.write(volume * samples)
    stream.stop_stream()
    stream.close()
    p.terminate()

# Initialize person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cap = cv2.VideoCapture(0)

capture_interval = 3  # seconds
photo_number = 0
last_capture_time = 0  # Initialize

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    # Detect people
    (rects, weights) = hog.detectMultiScale(frame, winStride=(8, 8),
                                            padding=(8, 8), scale=1.05)

    detected = len(rects) > 0

    for (x, y, w, h) in rects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    playTone()

    if detected:
        current_time = time.time()
        if current_time - last_capture_time >= capture_interval:
            capture_photo(frame, photo_number)
            photo_number += 1
            last_capture_time = current_time

    cv2.imshow("Figure Detect", frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()