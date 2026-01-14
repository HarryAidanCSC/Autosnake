import cv2 as cv
from cv2.typing import MatLike
import numpy as np
from pyscrcpy import Client
import time
import threading


class CropAdjuster:
    def __init__(self) -> None:
        """Construct a new Crop Adjuster."""
        self.current_frame = None
        self.client = None
        self.running = False

        # Default crop values
        self.left = 0
        self.right = 216
        self.top = 0
        self.bottom = 480

        # Colouring
        self.overlay_colour = (255, 0, 255)  # Magenta
        self.overlay_alpha = 0.4

    def start_video_feed(self) -> None:
        """Start live video feed from the device with high quality"""
        self.client = Client(
            max_size=480,
            bitrate=16000000,
            max_fps=20,
        )

        def on_frame(client: Client, frame: MatLike):
            """Function to set current frame.

            Args:
                client (Client): Connection to phone.
                frame (MatLike): Current frame.
            """
            if frame is not None:
                self.current_frame = frame

        self.client.on_frame(on_frame)
        self.running = True

        # Start in background thread
        def start_client():
            if self.client is not None:
                self.client.start(threaded=True)

        thread = threading.Thread(target=start_client, daemon=True)
        thread.start()

        # Wait for first frame
        start_time = time.time()
        while self.current_frame is None:
            if time.time() - start_time > 5:
                return
            time.sleep(0.01)

    def nothing(self, x):
        """Dummy callback for trackbars"""
        pass

    def run_interactive_adjuster(self):
        """Run interactive crop adjuster with live video feed"""
        if self.current_frame is None:
            print("No video feed available. Run start_video_feed() first.")
            return

        height, width = self.current_frame.shape[:2]  # type:ignore

        # Create windows
        cv.namedWindow("Live Feed with Crop Overlay")
        cv.namedWindow("Cropped View")
        cv.namedWindow("Crop Controls")

        # Create trackbars
        cv.createTrackbar("Left", "Crop Controls", self.left, width, self.nothing)
        cv.createTrackbar("Right", "Crop Controls", self.right, width, self.nothing)
        cv.createTrackbar("Top", "Crop Controls", self.top, height, self.nothing)
        cv.createTrackbar("Bottom", "Crop Controls", self.bottom, height, self.nothing)

        while self.running:
            if self.current_frame is None:
                time.sleep(0.01)
                continue

            # Get current frame
            frame = self.current_frame.copy()

            # Get current values
            left = cv.getTrackbarPos("Left", "Crop Controls")
            right = cv.getTrackbarPos("Right", "Crop Controls")
            top = cv.getTrackbarPos("Top", "Crop Controls")
            bottom = cv.getTrackbarPos("Bottom", "Crop Controls")

            # Ensure valid bounds
            if right <= left:
                right = left + 1
                cv.setTrackbarPos("Right", "Crop Controls", right)
            if bottom <= top:
                bottom = top + 1
                cv.setTrackbarPos("Bottom", "Crop Controls", bottom)

            # Clamp to frame boundaries
            right = min(right, width)
            bottom = min(bottom, height)

            # Create translucent magenta overlay
            overlay = frame.copy()

            # Draw filled rectangle in magenta
            cv.rectangle(overlay, (left, top), (right, bottom), self.overlay_colour, -1)

            # Blend overlay with original frame for transparency
            display_frame = cv.addWeighted(
                overlay, self.overlay_alpha, frame, 1 - self.overlay_alpha, 0
            )

            # Draw outline for clear boundary
            cv.rectangle(
                display_frame, (left, top), (right, bottom), self.overlay_colour, 1
            )

            # Add text overlay with current values
            crop_width = right - left
            crop_height = bottom - top

            # Show live feed with overlay
            cv.imshow("Live Feed with Crop Overlay", display_frame)

            # Show cropped region
            cropped = frame[top:bottom, left:right]
            if cropped.size > 0:
                # Resize in case cropped size is too small and crashes programme
                if crop_width < 300 or crop_height < 300:
                    scale = max(300 / crop_width, 300 / crop_height)
                    new_width = int(crop_width * scale)
                    new_height = int(crop_height * scale)
                    cropped_resized = cv.resize(cropped, (new_width, new_height))
                    cv.imshow("Cropped View", cropped_resized)
                else:
                    cv.imshow("Cropped View", cropped)

            # Control panel info
            control_panel = np.zeros((200, 400, 3), dtype=np.uint8)
            cv.putText(
                control_panel,
                "Crop Adjuster - LIVE",
                (10, 30),
                cv.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 255),
                2,
            )
            cv.putText(
                control_panel,
                "Press 's' to save values",
                (10, 80),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                1,
            )
            cv.putText(
                control_panel,
                "Press 'q' to quit",
                (10, 110),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                1,
            )
            cv.imshow("Crop Controls", control_panel)

            # Handle key presses
            key = cv.waitKey(1) & 0xFF

            if key == ord("q"):
                self.running = False
                break
            elif key == ord("s"):
                self.print_crop_values(left, right, top, bottom, width, height)

        # Cleanup
        if self.client and self.client.alive:
            self.client.stop()
        cv.destroyAllWindows()

    def print_crop_values(
        self,
        left: int,
        right: int,
        top: int,
        bottom: int,
        frame_width: int,
        frame_height: int,
    ) -> None:
        """Display crop values"""

        crop_width = right - left
        crop_height = bottom - top

        # Calculate percentages
        left_pct = (left / frame_width) * 100
        right_pct = (right / frame_width) * 100
        top_pct = (top / frame_height) * 100
        bottom_pct = (bottom / frame_height) * 100

        # Save to file
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


def main():

    adjuster = CropAdjuster()
    adjuster.start_video_feed()
    adjuster.run_interactive_adjuster()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
