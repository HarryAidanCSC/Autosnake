import ctypes


def fix_dpi_scaling():
    """
    Makes the process DPI aware.
    Must be called before interacting with the screen.
    """
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception as e:
        # Try to use system level awareness
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
            print("DPI awareness set to System Standard")
        except Exception as e:
            print(f"Could not set DPI awareness: {e}")
