from ImageAnalyser import ImageAnalyser
import time
from typing import Callable
from cv2.typing import MatLike
from pyscrcpy import Client
import cv2 as cv


class SetupMerchant:
    def __init__(
        self, client_config: dict[str, int], confidence_threshold: float = 0.8
    ) -> None:
        self.image_analyser = ImageAnalyser()
        self.CONFIDENCE_THRESHOLD = confidence_threshold

        # Define pixel ranges
        self.lpx, self.rpx, self.tpx, self.bpx = 0, 0, 0, 0
        self.cap_w, self.cap_h = 0, 0

        # Initalise client
        self.client = Client(
            max_size=client_config["max_size"],
            bitrate=client_config["bitrate"],
            max_fps=client_config["max_fps"],
        )

    def calibrate_capture_region(
        self, cog_file_path: str, restart_callback: Callable[[Client], None]
    ) -> None:
        """Calibrate existing capture region to JUST the relevant part of the screen

        Arguements:
            cog_file_path (str): File path of cog icon for template matching.
        Raises:
            RuntimeError: Screen grab could not be taken.
            RuntimeError: Could not find a valid capture region.
        """
        self.latest_frame = None  # Ensure we don't have old data
        self.last_action_time = 0

        # Read in cog image template
        self.cog_image = self.image_analyser._read_image(file_path=cog_file_path)

        # Temporary function that saves the frame
        def get_one_frame(client: Client, frame: MatLike) -> None:
            """Function to take one single frame during setup"

            Args:
                client (Client): Connection to client.
                frame (MatLike): Current viewing frame.
            """
            if frame is None:
                return

            # Exit if last action was recent
            if time.time() - self.last_action_time < 1.5:
                return
            # Check if the frame contains the setting cog icon
            value, _ = self.image_analyser.match_with_template(
                img=self.cog_image, comparison_frame=frame
            )

            # If the cog is in frame, press play
            if value > self.CONFIDENCE_THRESHOLD:
                restart_callback(client)
                self.last_action_time = time.time()
                return

            if frame is not None:
                self.latest_frame = frame
                self.client.stop()

        self.client.on_frame(get_one_frame)
        self.client.start(threaded=True)

        #  Wait until we have the first frame
        start_time = time.time()
        while self.latest_frame is None:
            # Emergency exit if it takes too long
            if time.time() - start_time > 5:
                raise RuntimeError("Could not capture an inital frame during setup.")
            time.sleep(0.01)

        # Find the capture region
        hsv_frame = cv.cvtColor(self.latest_frame, cv.COLOR_BGR2HSV)

        # Kill the client
        self.client.stop()

        # Colour filtering for playable region
        capture_region_loc = self.image_analyser.get_corners(
            frame=hsv_frame,
            lower_bound=[32, 148, 173],
            upper_bound=[46, 175, 223],
        )

        if capture_region_loc is None:
            raise RuntimeError("Could not find the initial start frame.")

        top_left, bottom_right = capture_region_loc

        # Calibrate new capture region
        self.lpx, self.tpx = top_left
        self.rpx, self.bpx = bottom_right

        self.cap_w, self.cap_h = (
            self.rpx - self.lpx,
            self.bpx - self.tpx,
        )
