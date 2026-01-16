import pygetwindow as gw
import keyboard
import time


def setup_chrome_macro(
    url: str = "google snake",
    delay: float = 0.1,
    browser_title: str = "Google Chrome",
) -> None:
    """Open a fresh chrome tab.

    Args:
        url (str, optional): Url to open. Defaults to "google snake".
        delay (float, optional): Default delay between actions in seconds. Defaults to 0.1.
        browser_title (str, optional): Name of the window. Defaults to "Google Chrome".

    Raises:
        RuntimeError: If Chrome is not open.
    """
    try:
        chrome_windows = gw.getWindowsWithTitle(browser_title)
        if not chrome_windows:
            raise RuntimeError(
                f"No {browser_title} windows found, check you have one open."
            )

        # Only require one window to play dino in
        chrome_window = chrome_windows[0]

        # Ensure window is maximised and accessible for the bot
        if chrome_window.isMinimized:
            chrome_window.restore()

        chrome_window.maximize()
        chrome_window.activate()

        time.sleep(delay)

        # Force Chrome to exit fullscreen (F11) mode
        start_height = chrome_window.height
        keyboard.press_and_release("f11")
        time.sleep(delay)

        # Measure and compare new height
        new_height = chrome_window.height
        if new_height > start_height:
            keyboard.press_and_release("f11")

        # Create new dino tab
        keyboard.press_and_release("ctrl + t")  # New tab
        keyboard.press_and_release("ctrl + l")  # Search bar
        keyboard.write(url)
        keyboard.press_and_release("enter")
        keyboard.press_and_release("ctrl + 0")  # Default 100% zoom

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
