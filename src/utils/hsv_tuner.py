import cv2 as cv
from typing import Any
import numpy as np
from masking import apply_morpho_masking


def nothing(x: Any) -> None:
    """Dummy callback for trackbars"""
    pass


def tune_hsv_colour(image_path: str) -> None:
    """
    Interactive HSV colour tuner

    Args:
        image_path: Path to a screenshot containing the snake
    """
    # Read the image
    frame = cv.imread(image_path)
    if frame is None:
        print(f"Error: Could not load image from {image_path}")
        return

    # Create window and trackbars
    cv.namedWindow("HSV Tuner")
    cv.namedWindow("Original")
    cv.namedWindow("Mask")
    cv.namedWindow("Result")

    # Initial HSV values
    HUE_MAX, SAT_MAX, VAL_MAX = 179, 255, 255
    cv.createTrackbar("H Min", "HSV Tuner", 0, HUE_MAX, nothing)
    cv.createTrackbar("H Max", "HSV Tuner", HUE_MAX, HUE_MAX, nothing)
    cv.createTrackbar("S Min", "HSV Tuner", 0, SAT_MAX, nothing)
    cv.createTrackbar("S Max", "HSV Tuner", SAT_MAX, SAT_MAX, nothing)
    cv.createTrackbar("V Min", "HSV Tuner", 0, VAL_MAX, nothing)
    cv.createTrackbar("V Max", "HSV Tuner", VAL_MAX, VAL_MAX, nothing)

    while True:
        # Get current trackbar positions
        h_min = cv.getTrackbarPos("H Min", "HSV Tuner")
        h_max = cv.getTrackbarPos("H Max", "HSV Tuner")
        s_min = cv.getTrackbarPos("S Min", "HSV Tuner")
        s_max = cv.getTrackbarPos("S Max", "HSV Tuner")
        v_min = cv.getTrackbarPos("V Min", "HSV Tuner")
        v_max = cv.getTrackbarPos("V Max", "HSV Tuner")

        # Convert to HSV
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        # Create mask
        lower_bound = np.array([h_min, s_min, v_min], dtype="uint8")
        upper_bound = np.array([h_max, s_max, v_max], dtype="uint8")
        mask = cv.inRange(hsv, lower_bound, upper_bound)

        # Apply morphological operations
        mask = apply_morpho_masking(mask=mask)

        # Apply mask to original image
        result = cv.bitwise_and(frame, frame, mask=mask)

        # Find contours and draw them
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        result_with_contours = result.copy()

        if contours:
            valid_contours = [c for c in contours if cv.contourArea(c) > 50]
            if valid_contours:
                largest = max(valid_contours, key=cv.contourArea)
                x, y, w, h = cv.boundingRect(largest)
                cv.rectangle(
                    result_with_contours, (x, y), (x + w, y + h), (0, 255, 0), 2
                )

                # Calculate and show centre
                M = cv.moments(largest)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv.circle(result_with_contours, (cx, cy), 5, (0, 0, 255), -1)

        # Display current HSV values on the tuner window
        info_img = np.zeros((150, 400, 3), dtype=np.uint8)
        cv.putText(
            info_img,
            f"Lower HSV: [{h_min}, {s_min}, {v_min}]",
            (10, 40),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )
        cv.putText(
            info_img,
            f"Upper HSV: [{h_max}, {s_max}, {v_max}]",
            (10, 80),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )
        cv.putText(
            info_img,
            "Press 'q' to quit, 's' to save",
            (10, 120),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

        # Show images
        cv.imshow("HSV Tuner", info_img)
        cv.imshow("Original", frame)
        cv.imshow("Mask", mask)
        cv.imshow("Result", result_with_contours)

        # Handle key presses
        key = cv.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            # Print to console wiht current config
            print("\n" + "=" * 50)
            print("Current HSV Values:")
            print(f"lower_bound=[{h_min}, {s_min}, {v_min}]")
            print(f"upper_bound=[{h_max}, {s_max}, {v_max}]")
            print("=" * 50 + "\n")

    cv.destroyAllWindows()
    print(f"\nFinal HSV Values:")
    print(f"lower_bound=[{h_min}, {s_min}, {v_min}]")
    print(f"upper_bound=[{h_max}, {s_max}, {v_max}]")


if __name__ == "__main__":
    image_path = "debug_screenshot.png"
    tune_hsv_colour(image_path)
