import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
from vision import UltrakillVision
from inputs import UltrakillInputs
import time

class UltrakillEnv(gym.Env):
    def __init__(self):
        super(UltrakillEnv, self).__init__()
        self.vision = UltrakillVision()
        self.inputs = UltrakillInputs()

        self.key_list = [
            'forward', 'left', 'back', 'right', 'jump', 'dash', 'slide',
            'punch', 'change_weapon', 'whiplash',
            'weapon_1', 'weapon_2', 'weapon_3', 'weapon_4', 'weapon_5', 'weapon_6'
        ]

        # Flattened Action Space:
        # 0-15: Keys (Binary)
        # 16: Left Click (0, 1, 2)
        # 17: Right Click (0, 1, 2)
        # 18-19: Mouse X, Y (Continuous -100 to 100)
        low = np.array([0]*18 + [-100, -100], dtype=np.float32)
        high = np.array([1]*16 + [2, 2] + [100, 100], dtype=np.float32)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.observation_space = spaces.Box(low=0, high=255, shape=(128, 128, 3), dtype=np.uint8)

        self.last_health = 100
        self.last_score = 0
        self.last_keys = [0] * 16
        self.last_clicks = [0, 0]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._release_all()
        frame = self.vision.capture_screen()
        obs = cv2.resize(frame, (128, 128))
        self.last_health = self.vision.get_health()
        self.last_score = self.vision.get_score()
        return obs, {}

    def step(self, action):
        self._apply_action_flat(action)

        frame = self.vision.capture_screen()
        obs = cv2.resize(frame, (128, 128))

        health = self.vision.get_health()
        score = self.vision.get_score()
        enemy_present = self.vision.detect_enemies(frame)

        reward = (score - self.last_score) * 0.1
        reward += (health - self.last_health) * 0.5
        if enemy_present:
            reward += 0.1

        self.last_health = health
        self.last_score = score

        terminated = health <= 0
        return obs, reward, terminated, False, {}

    def _apply_action_flat(self, action):
        # 0-15: Keys
        for i in range(16):
            val = 1 if action[i] > 0.5 else 0
            if val != self.last_keys[i]:
                if val == 1: self.inputs.hold_key(self.key_list[i])
                else: self.inputs.release_key(self.key_list[i])
                self.last_keys[i] = val

        # 16: Left Click
        l_click = int(np.round(action[16]))
        if l_click != self.last_clicks[0]:
            if l_click == 1: self.inputs.left_click()
            elif l_click == 2: self.inputs.hold_mouse('left')
            else: self.inputs.release_mouse('left')
            self.last_clicks[0] = l_click

        # 17: Right Click
        r_click = int(np.round(action[17]))
        if r_click != self.last_clicks[1]:
            if r_click == 1: self.inputs.right_click()
            elif r_click == 2: self.inputs.hold_mouse('right')
            else: self.inputs.release_mouse('right')
            self.last_clicks[1] = r_click

        # 18-19: Mouse Move
        self.inputs.mouse_move(int(action[18]), int(action[19]))

    def _release_all(self):
        for key in self.key_list:
            self.inputs.release_key(key)
        self.inputs.release_mouse('left')
        self.inputs.release_mouse('right')
        self.last_keys = [0] * 16
        self.last_clicks = [0, 0]
