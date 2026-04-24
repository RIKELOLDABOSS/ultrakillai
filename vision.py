import cv2
import mss
import numpy as np
import pytesseract
import time
import os

# Tesseract Configuration
TESSERACT_CMD = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
if os.path.exists(TESSERACT_CMD):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

class UltrakillVision:
    def __init__(self):
        self.sct = mss.mss()
        self.monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}

        # 1080p UI Regions
        self.health_roi = {"top": 930, "left": 50, "width": 200, "height": 80}
        self.score_roi = {"top": 50, "left": 1600, "width": 250, "height": 100}

    def capture_screen(self):
        img = np.array(self.sct.grab(self.monitor))
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

    def get_health(self):
        try:
            img = np.array(self.sct.grab(self.health_roi))
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            _, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)
            text = pytesseract.image_to_string(img, config='--psm 7 digits')
            return int(''.join(filter(str.isdigit, text)))
        except:
            return 100

    def get_score(self):
        try:
            img = np.array(self.sct.grab(self.score_roi))
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            _, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)
            text = pytesseract.image_to_string(img, config='--psm 7 digits')
            return int(''.join(filter(str.isdigit, text)))
        except:
            return 0

    def detect_enemies(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        lower_red = np.array([0, 100, 100])
        upper_red = np.array([10, 255, 255])
        mask = cv2.inRange(hsv, lower_red, upper_red)
        return np.sum(mask) > 5000
