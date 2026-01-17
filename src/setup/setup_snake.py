from setup.setup_chrome_macro import setup_chrome_macro
import time
from TheExtrovert import TheExtrovert


def setup_snake(ui: TheExtrovert):
    # Setup tab
    setup_chrome_macro()
    time.sleep(2)

    # Setup snake game
    ui.click_screen(x=0, y=480, apply_offset=True)
    time.sleep(3)

    # Full screen
    ui.press_keyboard("f")
    time.sleep(0.4)

    # Settings
    ui.click_screen(x=500, y=550, apply_offset=True)
    time.sleep(0.04)

    # Small Map
    ui.click_screen(x=420, y=240, apply_offset=True)
    time.sleep(0.04)

    # Slow
    ui.click_screen(x=480, y=210, apply_offset=True)
    time.sleep(0.04)

    # Mute
    ui.click_screen(x=740, y=-80, apply_offset=True)
    time.sleep(0.04)

    # Play
    ui.click_screen(coords_cache_key="restart")
