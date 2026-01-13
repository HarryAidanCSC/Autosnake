from adbutils import adb


class AndroidConnectionError(Exception):
    """Raised when the bot cannot connect to the phone."""

    pass


def check_device_connection() -> None:

    devices = adb.device_list()

    if not devices:
        raise AndroidConnectionError("Connection to phone could not be establised.")
