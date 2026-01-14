import cv2 as cv
import numpy as np
from pyscrcpy import Client
import time

# import utils.kill_button
import keyboard
from pyscrcpy import const
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from utils import kill_button

# Helper Mapping for Snake
KEYS = {
    "UP": const.KEYCODE_DPAD_UP,
    "DOWN": const.KEYCODE_DPAD_DOWN,
    "LEFT": const.KEYCODE_DPAD_LEFT,
    "RIGHT": const.KEYCODE_DPAD_RIGHT,
}

const.KEYCODE_MOVE_HOME


def move_snake(client, direction):
    """
    Instantly turns the snake.
    direction: "UP", "DOWN", "LEFT", "RIGHT"
    """
    code = KEYS.get(direction)
    if code:
        # Keycodes are fast (socket based)
        client.control.keycode(code, const.ACTION_DOWN)
        client.control.keycode(code, const.ACTION_UP)
    print("Action Done")


def on_frame(client, frame):
    """
    This function runs every time the phone sends a new video frame.
    frame: A numpy array representing the image (BGR colour)
    """
    if frame is None:
        print("na")
        return
    # 1. Convert frame to grayscale for matching (optional but recommended)
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # 6. Show the video feed with the overlay
    cv.imshow("Bot Vision", frame)
    cv.waitKey(1)

    key = cv.waitKey(1)
    if key == ord("s"):
        cv.imwrite("debug_screenshot.png", frame)
        print("Saved debug_screenshot.png! Crop your template from THIS file.")


if __name__ == "__main__":
    client = Client(max_size=480, bitrate=800000, max_fps=20)
    client.on_frame(on_frame)

    print("Starting... Press Ctrl+C to stop.")
    client.start(threaded=True)
    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        client.stop()
        cv.destroyAllWindows()
