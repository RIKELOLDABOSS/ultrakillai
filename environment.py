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

        self.action_space = spaces.Dict({
            "keys": spaces.MultiDiscrete([2] * 16),
            "mouse_click": spaces.MultiDiscrete([3, 3]),
            "mouse_move": spaces.Box(low=-100, high=100, shape=(2,), dtype=np.float32)
        })

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
        self._apply_action_delta(action)

        frame = self.vision.capture_screen()
        obs = cv2.resize(frame, (128, 128))

        health = self.vision.get_health()
        score = self.vision.get_score()

        reward = (score - self.last_score) * 0.1
        reward += (health - self.last_health) * 0.5

        self.last_health = health
        self.last_score = score

        terminated = health <= 0
        return obs, reward, terminated, False, {}

    def _apply_action_delta(self, action):
        # Only press/release if state changed to reduce lag
        for i, val in enumerate(action['keys']):
            if val != self.last_keys[i]:
                if val == 1: self.inputs.hold_key(self.key_list[i])
                else: self.inputs.release_key(self.key_list[i])
                self.last_keys[i] = val

        # Left Click
        l_click = action['mouse_click'][0]
        if l_click != self.last_clicks[0]:
            if l_click == 1: self.inputs.left_click()
            elif l_click == 2: self.inputs.hold_mouse('left')
            else: self.inputs.release_mouse('left')
            self.last_clicks[0] = l_click

        # Right Click
        r_click = action['mouse_click'][1]
        if r_click != self.last_clicks[1]:
            if r_click == 1: self.inputs.right_click()
            elif r_click == 2: self.inputs.hold_mouse('right')
            else: self.inputs.release_mouse('right')
            self.last_clicks[1] = r_click

        # Mouse Move
        self.inputs.mouse_move(int(action['mouse_move'][0]), int(action['mouse_move'][1]))

    def _release_all(self):
        for key in self.key_list:
            self.inputs.release_key(key)
        self.inputs.release_mouse('left')
        self.inputs.release_mouse('right')
        self.last_keys = [0] * 16
        self.last_clicks = [0, 0]
