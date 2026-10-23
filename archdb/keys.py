"""Decode terminal input before dispatching application shortcuts."""

from __future__ import annotations

import curses
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Mouse:
    x: int
    y: int
    action: str


ARROWS = {
    "A": curses.KEY_UP,
    "B": curses.KEY_DOWN,
    "C": curses.KEY_RIGHT,
    "D": curses.KEY_LEFT,
    "H": curses.KEY_HOME,
    "F": curses.KEY_END,
}


def _mouse(button: int, x: int, y: int, pressed: bool = True) -> Mouse | None:
    if not pressed:
        return None
    if button & 64:
        return Mouse(x, y, ("up", "down", "left", "right")[button & 3])
    return Mouse(x, y, "click") if button & 3 == 0 and not button & 32 else None


def read_key(screen):
    key = screen.get_wch()
    if key == curses.KEY_MOUSE:
        try:
            _, x, y, _, buttons = curses.getmouse()
        except curses.error:
            return None
        if buttons & curses.BUTTON4_PRESSED:
            return Mouse(x, y, "up")
        if buttons & getattr(curses, "BUTTON5_PRESSED", 0):
            return Mouse(x, y, "down")
        if buttons & (curses.BUTTON1_CLICKED | curses.BUTTON1_PRESSED | curses.BUTTON1_DOUBLE_CLICKED):
            return Mouse(x, y, "click")
        return None
    if isinstance(key, int):
        if key == getattr(curses, "KEY_SLEFT", -1):
            return curses.KEY_LEFT
        if key == getattr(curses, "KEY_SRIGHT", -1):
            return curses.KEY_RIGHT
        try:
            name = curses.keyname(key).decode()
        except curses.error:
            return key
        for prefix, direction in (
            ("kLFT", curses.KEY_LEFT),
            ("kRIT", curses.KEY_RIGHT),
            ("kUP", curses.KEY_UP),
            ("kDN", curses.KEY_DOWN),
        ):
            if name.startswith(prefix):
                return direction
        return key
    if key != "\x1b":
        return key
    # macOS terminals can send Esc+b/Esc+f for Option arrows. Consume the whole
    # event so its trailing 'f' never becomes the application's Filter shortcut.
    screen.timeout(15)
    try:
        try:
            next_key = screen.get_wch()
        except curses.error:
            return "\x1b"
        if next_key in ("b", "f"):
            return curses.KEY_LEFT if next_key == "b" else curses.KEY_RIGHT
        if next_key not in ("[", "O"):
            return next_key if isinstance(next_key, int) else None
        sequence = str(next_key)
        while len(sequence) < 64:
            try:
                part = screen.get_wch()
            except curses.error:
                return None
            if not isinstance(part, str):
                return None
            sequence += part
            if sequence == "[M":
                try:
                    button, x, y = (screen.get_wch() for _ in range(3))
                    return _mouse(ord(button) - 32, ord(x) - 33, ord(y) - 33)
                except (curses.error, TypeError):
                    return None
            if len(sequence) > 1 and "@" <= part <= "~":
                if match := re.fullmatch(r"\[<(\d+);(\d+);(\d+)([Mm])", sequence):
                    return _mouse(int(match[1]), int(match[2]) - 1, int(match[3]) - 1, match[4] == "M")
                if part in ARROWS:
                    return ARROWS[part]
                return {
                    "[5~": curses.KEY_PPAGE,
                    "[6~": curses.KEY_NPAGE,
                    "[3~": curses.KEY_DC,
                    "[Z": curses.KEY_BTAB,
                }.get(sequence)
        return None
    finally:
        screen.timeout(-1)
