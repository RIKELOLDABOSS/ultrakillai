import time
import keyboard
from pynput import mouse, keyboard as pynput_keyboard
import cv2
import mss
import numpy as np
import os
import pickle
import threading

class UltrakillTeacher:
    def __init__(self):
        self.sct = mss.mss()
        self.monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}
        self.recording = False
        self.data = []

        self.current_keys = set()
        self.mouse_pos = (0, 0)
        self.last_mouse_pos = (0, 0)
        self.mouse_buttons = {'left': False, 'right': False}
        self.mouse_delta = (0, 0)

    def on_press(self, key):
        try:
            k = key.char if hasattr(key, 'char') else key.name
            self.current_keys.add(k)
        except AttributeError:
            pass

    def on_release(self, key):
        try:
            k = key.char if hasattr(key, 'char') else key.name
            if k in self.current_keys:
                self.current_keys.remove(k)
        except AttributeError:
            pass

    def on_move(self, x, y):
        self.mouse_pos = (x, y)

    def on_click(self, x, y, button, pressed):
        if button == mouse.Button.left:
            self.mouse_buttons['left'] = pressed
        elif button == mouse.Button.right:
            self.mouse_buttons['right'] = pressed

    def start_recording(self):
        print("Recording started! Press 'y' again to stop.")
        self.recording = True
        self.data = []

        # Start Listeners
        self.k_listener = pynput_keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.m_listener = mouse.Listener(on_move=self.on_move, on_click=self.on_click)
        self.k_listener.start()
        self.m_listener.start()

        self.last_mouse_pos = self.mouse_pos

        while self.recording:
            if keyboard.is_pressed('y'):
                time.sleep(0.5)
                self.stop_recording()
                break

            # Capture state
            frame = np.array(self.sct.grab(self.monitor))
            frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB), (128, 128))

            # Calculate mouse delta
            dx = self.mouse_pos[0] - self.last_mouse_pos[0]
            dy = self.mouse_pos[1] - self.last_mouse_pos[1]
            self.last_mouse_pos = self.mouse_pos

            inputs = {
                'keys': list(self.current_keys),
                'mouse_click': [self.mouse_buttons['left'], self.mouse_buttons['right']],
                'mouse_move': (dx, dy)
            }

            self.data.append({'frame': frame, 'inputs': inputs})
            time.sleep(0.05) # 20 FPS recording

    def stop_recording(self):
        self.recording = False
        self.k_listener.stop()
        self.m_listener.stop()
        print(f"Recording stopped. Captured {len(self.data)} frames.")
        self.save_data()

    def save_data(self):
        if not os.path.exists('recordings'):
            os.makedirs('recordings')
        filename = f"recordings/demo_{int(time.time())}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(self.data, f)
        print(f"Data saved to {filename}")

if __name__ == "__main__":
    teacher = UltrakillTeacher()
    print("Press 'y' to start recording your gameplay.")
    while True:
        if keyboard.is_pressed('y'):
            time.sleep(0.5)
            teacher.start_recording()
            break
        time.sleep(0.1)
