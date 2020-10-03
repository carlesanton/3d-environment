import cv2
import time

web_cam = cv2.VideoCapture(0)

while True:
    start_time = time.time()
    image = web_cam.read()
    d_t = time.time() - start_time
    start_time = time.time()

    frame_rate = 1/(d_t)
    print(f'Frame rate: {frame_rate} fps')