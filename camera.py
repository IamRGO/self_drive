import cv2
import serial

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
camera.set(cv2.CAP_PROP_FPS, 30)

arduino = serial.Serial(port='/dev/ttyUSB0', baudrate=19200, timeout=5)

while True:
    _, frame = camera.read()  # read the camera frame