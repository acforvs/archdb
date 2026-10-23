import curses
from collections import deque

import pytest

from archdb.keys import Mouse, read_key


class Input:
    def __init__(self, values):
        self.values = deque(values)
        self.wait = -1

    def get_wch(self):
        if not self.values:
            raise curses.error("no input")
        return self.values.popleft()

    def timeout(self, value):
        self.wait = value


@pytest.mark.parametrize(
    "sequence,direction",
    [
        ("\x1bb", curses.KEY_LEFT),
        ("\x1bf", curses.KEY_RIGHT),
        ("\x1b[1;3D", curses.KEY_LEFT),
        ("\x1b[1;3C", curses.KEY_RIGHT),
        ("\x1b[1;9D", curses.KEY_LEFT),
        ("\x1b[1;9C", curses.KEY_RIGHT),
        ("\x1bOD", curses.KEY_LEFT),
        ("\x1bOC", curses.KEY_RIGHT),
    ],
)
def test_option_arrows_are_single_navigation_events(sequence, direction):
    screen = Input(sequence)
    assert read_key(screen) == direction
    assert not screen.values
    assert screen.wait == -1


@pytest.mark.parametrize(
    "button,action", [(64, "up"), (65, "down"), (66, "left"), (67, "right"), (0, "click")]
)
def test_sgr_mouse_reports_are_consumed_as_one_event(button, action):
    screen = Input(f"\x1b[<{button};90;18M")
    assert read_key(screen) == Mouse(89, 17, action)
    assert not screen.values


def test_legacy_mouse_reports_consume_coordinates_that_look_like_shortcuts():
    screen = Input("\x1b[Maof")
    assert read_key(screen) == Mouse(ord("o") - 33, ord("f") - 33, "down")
    assert not screen.values


def test_mouse_release_does_not_repeat_a_click():
    assert read_key(Input("\x1b[<0;90;18m")) is None


def test_bare_escape_and_unknown_alt_shortcuts_do_not_leak_letters():
    assert read_key(Input("\x1b")) == "\x1b"
    screen = Input("\x1be")
    assert read_key(screen) is None
    assert not screen.values
