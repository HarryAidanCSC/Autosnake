import cv2 as cv
import numpy as np
import mss
import time
import threading
from typing import Any
import traceback


class CropAdjuster:
    def __init__(self, monitor_num: int = 2) -> None:
        """Construct a new Crop Adjuster."""
        self.current_frame = None
        self.running = False
        self.lock = threading.Lock()

        # Default crop values
        with mss.mss() as sct:
            if monitor_num < len(sct.monitors):
                self.mon = sct.monitors[monitor_num]
            else:
                print(f"Monitor {monitor_num} not found, defaulting to 1")
                self.mon = sct.monitors[1]

        # Colouring
        self.overlay_colour = (255, 0, 255)
        self.overlay_alpha = 0.4
        self.buttons = self._create_buttons()

    def _create_buttons(self) -> list[dict[str, Any]]:
        """Generates the configuration for +/- buttons."""
        buttons = []
        start_y = 60
        gap_y = 45

        # Create a button for each crop dimensions
        for i, name in enumerate(["Left", "Right", "Top", "Bottom"]):
            y_pos = start_y + (i * gap_y)
            base = {"y": y_pos, "w": 30, "h": 30, "trackbar": name}

            # Minus
            buttons.append(
                {
                    **base,
                    "x": 180,
                    "text": "-",
                    "delta": -1,
                    "color": (0, 0, 255),
                }
            )

            # Plus
            buttons.append(
                {
                    **base,
                    "x": 220,
                    "text": "+",
                    "delta": 1,
                    "color": (0, 255, 0),
                }
            )
        return buttons

    def _capture_loop(self):
        """Background thread to keep grabbing the screen."""
        with mss.mss() as sct:
            while self.running:
                img = np.array(sct.grab(self.mon))
                img = cv.cvtColor(img, cv.COLOR_BGRA2BGR)

                with self.lock:
                    self.current_frame = img

                time.sleep(0.02)

    def start_video_feed(self) -> None:
        """Starts the background thread for screen capturing."""
        self.running = True
        t = threading.Thread(target=self._capture_loop, daemon=True)
        t.start()
        print("Video feed started...")

        while self.current_frame is None:
            time.sleep(0.1)

    @staticmethod
    def nothing(x):
        pass

    def adjust_trackbar(self, name: str, delta: int, max_val: int) -> None:
        """Increments/decrements a trackbar"""
        current = cv.getTrackbarPos(name, "Crop Controls")
        new_val = np.clip(current + delta, 0, max_val)
        cv.setTrackbarPos(name, "Crop Controls", new_val)

    def handle_clicks(self, event, x, y, flags, param):
        """Mouse Callback to handle button clicks"""
        if event != cv.EVENT_LBUTTONDOWN or self.current_frame is None:
            return

        for btn in self.buttons:
            bx, by, bw, bh = btn["x"], btn["y"], btn["w"], btn["h"]
            if bx <= x <= bx + bw and by <= y <= by + bh:
                is_width = btn["trackbar"] in ["Left", "Right"]
                max_val = (
                    self.current_frame.shape[1]
                    if is_width
                    else self.current_frame.shape[0]
                )
                self.adjust_trackbar(btn["trackbar"], btn["delta"], max_val)

    def _show_cropped_view(self, frame, coords):
        """Displays the cropped region, resized if too small"""
        left, right, top, bottom = coords
        cropped = frame[top:bottom, left:right]

        if cropped.size == 0:
            return

        h, w = cropped.shape[:2]
        if w < 300 or h < 300:
            scale = max(300 / w, 300 / h) if w > 0 and h > 0 else 1
            cropped = cv.resize(cropped, None, fx=scale, fy=scale)

        cv.imshow("Cropped View", cropped)

    def _draw_control_panel(self, values: dict) -> np.ndarray:
        """Draws the control panel with buttons and values"""
        panel = np.zeros((300, 400, 3), dtype=np.uint8)

        # Draw Header
        cv.putText(
            panel,
            "PRECISION CONTROLS",
            (10, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # Draw Labels and Current Values
        for i, (name, val) in enumerate(values.items()):
            y_text = 60 + (i * 45) + 25
            cv.putText(
                panel,
                f"{name}: {val}",
                (20, y_text),
                cv.FONT_HERSHEY_PLAIN,
                1.5,
                (200, 200, 200),
                2,
            )

        # Draw Buttons
        for btn in self.buttons:
            bx, by, bw, bh = btn["x"], btn["y"], btn["w"], btn["h"]
            cv.rectangle(panel, (bx, by), (bx + bw, by + bh), btn["color"], -1)

            # Center text in button
            text_size = cv.getTextSize(btn["text"], cv.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            text_x = bx + (bw - text_size[0]) // 2
            text_y = by + (bh + text_size[1]) // 2
            cv.putText(
                panel,
                btn["text"],
                (text_x, text_y),
                cv.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 0),
                2,
            )

        # Footer
        cv.putText(
            panel,
            "S: Print Config | Q: Quit",
            (10, 280),
            cv.FONT_HERSHEY_PLAIN,
            1.1,
            (100, 255, 100),
            1,
        )
        return panel

    def run_interactive_adjuster(self):
        """Run interactive crop adjuster with live video feed"""
        if self.current_frame is None:
            print("No video feed. Run start_video_feed() first.")
            return

        with self.lock:
            height, width = self.current_frame.shape[:2]

        # Create windows
        cv.namedWindow("Live Feed with Crop Overlay")
        cv.namedWindow("Cropped View")
        cv.namedWindow("Crop Controls")
        cv.setMouseCallback("Crop Controls", self.handle_clicks)

        # Create trackbars
        for name, max_dim in [
            ("Left", width),
            ("Right", width),
            ("Top", height),
            ("Bottom", height),
        ]:
            cv.createTrackbar(
                name,
                "Crop Controls",
                0 if "Left" in name or "Top" in name else max_dim,
                max_dim,
                self.nothing,
            )

        print("Press 'q' to quit, 's' to save/print values.")

        while self.running:
            with self.lock:
                if self.current_frame is None:
                    continue
                frame = self.current_frame.copy()

            height, width = frame.shape[:2]

            # 1. Read Trackbars
            names = ["Left", "Right", "Top", "Bottom"]
            values = {name: cv.getTrackbarPos(name, "Crop Controls") for name in names}

            # 2. Logic to prevent negative crops
            if values["Right"] <= values["Left"]:
                values["Right"] = values["Left"] + 1
                cv.setTrackbarPos("Right", "Crop Controls", values["Right"])
            if values["Bottom"] <= values["Top"]:
                values["Bottom"] = values["Top"] + 1
                cv.setTrackbarPos("Bottom", "Crop Controls", values["Bottom"])

            # Clamp boundaries
            values["Right"] = min(values["Right"], width)
            values["Bottom"] = min(values["Bottom"], height)

            # 3. Draw Overlay
            overlay = frame.copy()
            p1 = (values["Left"], values["Top"])
            p2 = (values["Right"], values["Bottom"])

            cv.rectangle(overlay, p1, p2, self.overlay_colour, -1)
            display_frame = cv.addWeighted(
                overlay, self.overlay_alpha, frame, 1 - self.overlay_alpha, 0
            )
            cv.rectangle(display_frame, p1, p2, self.overlay_colour, 1)
            cv.imshow("Live Feed with Crop Overlay", display_frame)

            # 4. Show Cropped View
            self._show_cropped_view(
                frame,
                (values["Left"], values["Right"], values["Top"], values["Bottom"]),
            )

            # 5. Draw Control Panel
            panel = self._draw_control_panel(values)
            cv.imshow("Crop Controls", panel)

            # Handle key presses
            key = cv.waitKey(1) & 0xFF
            if key == ord("q"):
                self.running = False
                break
            elif key == ord("s"):
                self.print_crop_values(values, width, height)

        cv.destroyAllWindows()

    def print_crop_values(
        self, values: dict, frame_width: int, frame_height: int
    ) -> None:
        """Display crop values formatted for YAML/Config"""
        left = values["Left"]
        right = values["Right"]
        top = values["Top"]
        bottom = values["Bottom"]

        crop_width = right - left
        crop_height = bottom - top

        # Calculate percentages
        left_pct = (left / frame_width) * 100
        right_pct = (right / frame_width) * 100
        top_pct = (top / frame_height) * 100
        bottom_pct = (bottom / frame_height) * 100

        # Save to file
        print("\n" + "=" * 30)
        print("REPLACE_WITH_CONFIG_NAME:")
        print("  cell_cols: REPLACE")
        print("  cell_rows: REPLACE")
        # Frame sizing
        print("  frame_size:")
        print(f"    width: {frame_width}")
        print(f"    height: {frame_height}")
        # Cropped sizing
        print("  crop_size:")
        print(f"    width: {crop_width}")
        print(f"    height: {crop_height}")
        # Absolute pixels
        print(f"  absolute_values:")
        print(f"    lpx: {left}")
        print(f"    rpx: {right}")
        print(f"    tpx: {top}")
        print(f"    bpx: {bottom}")
        # Relative positions
        print(f"  percs:")
        print(f"    left: {left_pct:.1f}")
        print(f"    right: {right_pct:.1f}")
        print(f"    top: {top_pct:.1f}")
        print(f"    bottom: {bottom_pct:.1f}")
        print("=" * 30 + "\n")


def main():
    # Adjust monitor_num if needed
    adjuster = CropAdjuster(monitor_num=2)
    adjuster.start_video_feed()
    adjuster.run_interactive_adjuster()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
