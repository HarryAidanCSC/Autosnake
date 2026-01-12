import os
import keyboard


def kill_button() -> None:
    """Keyboard escape if mouse cannot be force quitted."""
    print("\nEscape Button Pressed. Force quitting...")
    os._exit(0)


# Run hotkey in background
keyboard.add_hotkey("esc", kill_button)
