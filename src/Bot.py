# Third-Party
import cv2 as cv
import numpy as np
import time
import keyboard

# Type Hints
from typing import Tuple, Optional
from cv2.typing import MatLike

# Custom
from solve_grid import breadth_first_search
from setup.setup_snake import setup_snake
from utils.kill_button import kill_button
from ImageAnalyser import ImageAnalyser
from GameMap import GameMap
from FrameMerchant import FrameMerchant
from SnakeState import SnakeState
from TheExtrovert import TheExtrovert


class Bot:
    """Bot to automate Snake."""

    def __init__(
        self,
        let_bot_take_control: bool,
        config_name: str,
        setup_snake_from_scratch: bool = True,
        visual_debug: bool = True,
    ):
        """Construct a new Bot to play Snake.

        Args:
            let_bot_take_control (bool): Give control of the game to the bot.
            map_config (str): Custom YAML name of map type.
            setup_snake_from_scratch (bool, optional): Set up brand new tab and snake settings. Defaults to True.
            visual_debug (bool, optional): Apply visual debugging overlay. Defaults to True.
        """

        # CONSTANTS
        self.BOT_CAN_PLAY_AUTONOMOUSLY = let_bot_take_control

        self.ui = TheExtrovert(config_name=config_name)
        self.snake_state = SnakeState(
            nrows=self.ui.display_config["cell_rows"],
            ncols=self.ui.display_config["cell_cols"],
        )
        # Start the snake game
        if setup_snake_from_scratch:
            setup_snake(ui=self.ui)

        # Initialise image calibration
        self.image_analyser = ImageAnalyser()

        # Run hotkey in background
        keyboard.add_hotkey("esc", kill_button)

        # Calculate width of each cell
        self.frame_merchant = FrameMerchant(
            cfg=self.ui.get_display_config(),
            debug_overlay=visual_debug,
        )

        # Initialise game components
        self.game_map = GameMap(
            grid_nc_nr=self.frame_merchant.get_grid_dims(),
            grid_hpx_wpx=self.frame_merchant.get_cell_dims(),
            state=self.snake_state,
        )

        # Movement settings
        self.MOVE_COOLDOWN_SECONDS = 0.05  # Minimum time between moves

    def play_snake(self) -> None:
        """Entry point to play a snake repeatedly."""

        time_of_last_screenshot = 0
        while True:
            # Break condition
            if keyboard.is_pressed("z"):
                break

            # CV Loop
            frame = self.ui.get_raw_frame()
            self._execute_strategy(frame=frame)
            self.ui.display_live_feed(frames={"Snake": frame})

            if keyboard.is_pressed("x") and time.time() - time_of_last_screenshot > 1.5:
                self.ui.save_screenshot(frame=frame)
                time_of_last_screenshot = time.time()
                print("Screenshot!")

    def _detect_and_process_object(
        self,
        hsv_frame: np.ndarray,
        frame: np.ndarray,
        lower_bound: list[int],
        upper_bound: list[int],
        colour_key: str,
        is_snake_head: bool = False,
    ) -> Optional[Tuple[int, int]]:
        """Detect an object using color filtering and return its center pixel coordinates.

        Args:
            hsv_frame (np.ndarray): HSV color space frame for filtering
            frame (np.ndarray): Original frame for drawing debug overlay
            lower_bound (list[int]): Lower HSV bounds for color filtering
            upper_bound (list[int]): Upper HSV bounds for color filtering
            colour_key (str): Color key for debug visualization
            is_snake_head (bool): Whether this is detecting the snake head

        Returns:
            Optional[Tuple[int, int]]: Pixel coordinates (x, y) or None if not found
        """
        obj_loc = self.image_analyser.colour_filtering(
            frame=hsv_frame,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            is_snake_head=is_snake_head,
        )
        if obj_loc is None:
            return None

        cx, cy, obj_w, obj_h = obj_loc

        # Convert center to top-left corner
        top_left_x = cx - (obj_w // 2)
        top_left_y = cy - (obj_h // 2)
        top_left = (top_left_x, top_left_y)

        # Draw debug overlay
        self.frame_merchant._write_box_on_frame(
            frame,
            top_left=top_left,
            width=obj_w,
            height=obj_h,
            colour_key=colour_key,
        )

        return (cx, cy)

    def _execute_strategy(self, frame: MatLike) -> None:
        # Exit if not configured correctly
        if frame is None:
            return

        # If snake is not moving then restart
        if (
            time.time() - self.snake_state.time_of_last_move > 1.5
            and self.BOT_CAN_PLAY_AUTONOMOUSLY
        ):
            self.snake_state.time_of_last_move = time.time()
            self.ui.press_keyboard("space")
            time.sleep(0.2)
            self.ui.press_keyboard("right")
            self.snake_state.reset_snake_state()

        # Convert frame to HSV for contour detection
        hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        # Detect snake head (returns PIXEL coords now)
        snake_pixel_pos = self._detect_and_process_object(
            hsv_frame=hsv_frame,
            frame=frame,
            lower_bound=[110, 166, 0],
            upper_bound=[112, 172, 240],
            colour_key="snakeHead",
            is_snake_head=True,
        )
        # Detect apple (returns PIXEL coords now)
        apl_pixel_pos = self._detect_and_process_object(
            hsv_frame=hsv_frame,
            frame=frame,
            lower_bound=[0, 140, 216],
            upper_bound=[8, 255, 252],
            colour_key="apple",
            is_snake_head=False,
        )

        if snake_pixel_pos and apl_pixel_pos:
            sx, sy = snake_pixel_pos
            direction = self.snake_state.cur_dir
            OFFSET = 35
            
            if direction == "dn":
                sy -= OFFSET
            elif direction == "ds":
                sy += OFFSET
            elif direction == "de":
                sx += OFFSET
            elif direction == "dw":
                sx -= OFFSET

            # Convert to Grid Coordinates
            snake_snooter_tip = self.frame_merchant._pixel_to_coords(
                x=sx, y=sy
            )

            ax, ay = apl_pixel_pos
            
            self.snake_state.apple_pos = self.frame_merchant._pixel_to_coords(
                x=ax, y=ay
            )

            # Don't pathfind if still in the same square
            if snake_snooter_tip != self.snake_state.snake_snoot_coords:
                self.snake_state.snake_snoot_coords = snake_snooter_tip
                self._on_new_cell(frame=frame)

        # Render Everything
        self._render_soul(frame=frame)

    def _on_new_cell(
        self,
        frame: MatLike,
    ) -> None:
        # Pathfinding
        self.game_map.build_grid(frame=frame)

        path = breadth_first_search(state=self.snake_state)
        if path:
            self.snake_state.cur_path = path
            self._move_snake()

    def _move_snake(self) -> None:

        if not self.BOT_CAN_PLAY_AUTONOMOUSLY:
            return

        # Rate limiting: check if enough time has passed since last move
        time_since_last_move = time.time() - self.snake_state.time_of_last_move
        if time_since_last_move < self.MOVE_COOLDOWN_SECONDS:
            return  # Too soon to move again

        if (
            self.snake_state.snake_snoot_coords[0] < 0
            or self.snake_state.snake_snoot_coords[1] < 0
        ):
            return


        dx = self.snake_state.cur_path[0][0] - self.snake_state.snake_snoot_coords[0]
        dy = self.snake_state.cur_path[0][1] - self.snake_state.snake_snoot_coords[1]

        # Which direction to move in
        new_dir, key = self.snake_state.dirs_alt[(dx, dy)]
        opposite_from_cur = self.snake_state.opposite_dir.get(self.snake_state.cur_dir)

        if new_dir == opposite_from_cur:
            return

        # Update current direction and timestamp
        self.ui.press_keyboard(keys=key)
        self.snake_state.cur_dir = new_dir
        self.snake_state.time_of_last_move = time.time()


    def _get_snake_snooter_tip(
        self, snake_head_pos: Tuple[int, int]
    ) -> Tuple[int, int]:
        """Calculates nothing, kept for legacy compatibility if needed."""
        x, y = snake_head_pos
        return x, y

    def _render_soul(self, frame: MatLike) -> None:
        self.frame_merchant.render_multi_coordinates(
            frame=frame,
            coordinates=self.snake_state.snake_body,
            colour_key="snake",
            is_grid_coords=False,
        )

        if self.snake_state.cur_path:
            self.frame_merchant.render_multi_coordinates(
                frame=frame,
                coordinates=self.snake_state.cur_path,
                colour_key="path",
                is_grid_coords=True,
            )

        if self.snake_state.snake_snoot_coords[0] >= 0:
            self.frame_merchant.render_multi_coordinates(
                frame=frame,
                coordinates=[self.snake_state.snake_snoot_coords],
                colour_key="snoot",
                is_grid_coords=True,
            )


if __name__ == "__main__":
    bot = Bot(
        let_bot_take_control=True,
        config_name="small",
        setup_snake_from_scratch=False,
        visual_debug=True,
    )

    # Start playing
    bot.play_snake()
