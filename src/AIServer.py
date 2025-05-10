import traceback
import os
import sys


def set_quick_edit_mode(turn_on=None) -> bool:
    """Enable/Disable windows console Quick Edit Mode"""
    import win32console  # pyright: ignore[reportMissingModuleSource]

    ENABLE_QUICK_EDIT_MODE = 0x40
    ENABLE_EXTENDED_FLAGS = 0x80

    screen_buffer = win32console.GetStdHandle(-10)
    orig_mode = screen_buffer.GetConsoleMode()
    is_on = orig_mode & ENABLE_QUICK_EDIT_MODE
    if is_on != turn_on and turn_on is not None:
        if turn_on:
            new_mode = orig_mode | ENABLE_QUICK_EDIT_MODE
        else:
            new_mode = orig_mode & ~ENABLE_QUICK_EDIT_MODE
        screen_buffer.SetConsoleMode(new_mode | ENABLE_EXTENDED_FLAGS)

    return is_on if turn_on is None else turn_on


if __name__ == "__main__":
    if os.name == "nt" and sys.stdout.isatty():
        set_quick_edit_mode(False)

    try:
        import AIServerInternal

        AIServerInternal.main()
    except Exception as e:
        print(e)
        print(traceback.format_exc())
        print("Press Enter to quit...")
        input()
        exit(1)
