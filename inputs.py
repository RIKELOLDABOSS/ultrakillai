import pydirectinput
import time

# Disable fail-safe to prevent the script from stopping if the mouse moves to a corner
# However, for safety during development, we might keep it or use a specific kill switch.
pydirectinput.FAILSAFE = False

class UltrakillInputs:
    def __init__(self):
        # Mapping for clarity
        self.keys = {
            'forward': 'w',
            'left': 'a',
            'back': 's',
            'right': 'd',
            'jump': 'space',
            'dash': 'shift',
            'slide': 'ctrl',
            'punch': 'e',
            'change_weapon': 'q',
            'whiplash': 'r',
            'weapon_1': '1',
            'weapon_2': '2',
            'weapon_3': '3',
            'weapon_4': '4',
            'weapon_5': '5',
            'weapon_6': '6',
        }

    def press_key(self, key_name):
        if key_name in self.keys:
            pydirectinput.press(self.keys[key_name])
        else:
            pydirectinput.press(key_name)

    def hold_key(self, key_name):
        key = self.keys.get(key_name, key_name)
        pydirectinput.keyDown(key)

    def release_key(self, key_name):
        key = self.keys.get(key_name, key_name)
        pydirectinput.keyUp(key)

    def mouse_move(self, x, y):
        """Move mouse relatively by x, y pixels."""
        pydirectinput.moveRel(x, y, relative=True)

    def left_click(self, duration=0):
        if duration > 0:
            pydirectinput.mouseDown(button='left')
            time.sleep(duration)
            pydirectinput.mouseUp(button='left')
        else:
            pydirectinput.click(button='left')

    def right_click(self, duration=0):
        if duration > 0:
            pydirectinput.mouseDown(button='right')
            time.sleep(duration)
            pydirectinput.mouseUp(button='right')
        else:
            pydirectinput.click(button='right')

    def hold_mouse(self, button='left'):
        pydirectinput.mouseDown(button=button)

    def release_mouse(self, button='left'):
        pydirectinput.mouseUp(button=button)

if __name__ == "__main__":
    # Small test
    print("Starting input test in 3 seconds...")
    time.sleep(3)
    inputs = UltrakillInputs()
    inputs.press_key('forward')
    print("Test complete.")
