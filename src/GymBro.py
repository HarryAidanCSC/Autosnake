from pyscrcpy import Client, const
from typing import Tuple


class GymBro:
    """Handles all client control operations."""

    def __init__(self, client: Client, device_dimensions: Tuple[int, int]):
        """Construct GymBro with client connection.

        Args:
            client: The pyscrcpy Client for device control
            device_dimensions: (height, width) of the device screen in pixels
        """
        self.client = client
        self.device_height, self.device_width = device_dimensions

        # Define movement keycodes
        self.KEYS = {
            "dn": const.KEYCODE_DPAD_UP,  # North
            "ds": const.KEYCODE_DPAD_DOWN,  # South
            "dw": const.KEYCODE_DPAD_LEFT,  # West
            "de": const.KEYCODE_DPAD_RIGHT,  # East
        }

    def send_direction(self, direction: str) -> None:
        """Send a directional movement command.

        Args:
            direction: Direction string ("dn", "ds", "de", "dw")
        """
        code = self.KEYS.get(direction)
        if code:
            self.client.control.keycode(code, const.ACTION_DOWN)
            self.client.control.keycode(code, const.ACTION_UP)

    def restart_game(self) -> None:
        """Tap the restart button to start a new game."""
        x_coord = int(self.device_width * 0.463)
        y_coord = int(self.device_height * 0.701)
        self.client.control.touch(x_coord, y_coord)

    def tap_screen(self, x_percent: float, y_percent: float) -> None:
        """Tap the screen at a percentage-based position.

        Args:
            x_percent: X percentage (0.0 to 1.0)
            y_percent: Y percentage (0.0 to 1.0)
        """
        x_coord = int(self.device_width * x_percent)
        y_coord = int(self.device_height * y_percent)
        self.client.control.touch(x_coord, y_coord)

    def handle_manual_input(
        self, key: str, current_direction: str, opposite_dir_map: dict
    ) -> Tuple[str, bool]:
        """Process manual keyboard input from user.

        Args:
            key: Key pressed ("w", "a", "s", "d", "e")
            current_direction: Current snake direction
            opposite_dir_map: Mapping of opposite directions

        Returns:
            Tuple of (new_direction, restart_requested)
        """
        possible_keys = {"w", "a", "s", "d", "e"}

        if key not in possible_keys:
            return current_direction, False

        # Restart requested
        if key == "e":
            self.restart_game()
            return "de", True  # Reset to East

        # Map WASD to direction
        key_to_dir = {
            "w": "dn",
            "a": "dw",
            "s": "ds",
            "d": "de",
        }

        new_dir = key_to_dir.get(key)
        if not new_dir:
            return current_direction, False

        # Check for 180-degree turn
        opposite = opposite_dir_map.get(current_direction)
        if new_dir == opposite:
            return current_direction, False  # Block opposite movement

        # Send the movement command
        self.send_direction(new_dir)
        return new_dir, False
