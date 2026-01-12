from Bot import Bot

if __name__ == "__main__":
    bot = Bot(
        snake_file_path="assets/snake_eyes.png",
        apple_file_path="assets/apple.png",
        max_size=480,
        bitrate=800000,
        max_fps=20,
    )
    bot.play_snake()
