"""Streaming Qwen-style <think>...</think> reasoning splitter.

The generator emits arbitrary text fragments; markers may straddle fragment
boundaries. This splitter never invents reasoning: with no markers the visible
channel carries the raw text unchanged. Unknown templates stay conservative:
only the exact ``<think>``/``</think>`` pair is recognized.
"""

from __future__ import annotations

OPEN = "<think>"
CLOSE = "</think>"


def has_unclosed_think(text: str) -> bool:
    """True when the last ``<think>`` has no closing marker after it."""
    return text.rstrip().endswith(OPEN)


def split_complete(text: str, prefix_open: bool = False) -> tuple[str, str]:
    """Split buffered text into (reasoning, visible).

    Only the first ``<think>...</think>`` block is treated as reasoning; any
    further markers stay raw in the visible channel. Truncated thinking (open
    without close) puts the tail in reasoning and the head in visible.
    """
    splitter = ReasoningSplitter(prefix_open=prefix_open)
    reasoning, visible = splitter.feed(text)
    tail_reasoning, tail_visible = splitter.flush()
    return reasoning + tail_reasoning, visible + tail_visible


class ReasoningSplitter:
    """Incremental splitter for streamed fragments.

    ``prefix_open`` continues reasoning started by an unclosed ``<think>`` at
    the prompt suffix. ``feed`` returns (reasoning_delta, visible_delta);
    ``flush`` drains the tail held back for split-marker detection.
    """

    def __init__(self, prefix_open: bool = False) -> None:
        self.in_reasoning = prefix_open
        self._finished_reasoning = False
        self._buffer = ""

    def feed(self, text: str) -> tuple[str, str]:
        if not text:
            return "", ""
        self._buffer += text
        reasoning: list[str] = []
        visible: list[str] = []
        while self._buffer:
            if self._finished_reasoning:
                visible.append(self._buffer)
                self._buffer = ""
                break
            if self.in_reasoning:
                closed = self._buffer.find(CLOSE)
                if closed < 0:
                    # Hold back a CLOSE-length tail: it may be a split marker.
                    hold = min(len(self._buffer), len(CLOSE) - 1)
                    cut = len(self._buffer) - hold
                    reasoning.append(self._buffer[:cut])
                    self._buffer = self._buffer[cut:]
                    break
                reasoning.append(self._buffer[:closed])
                self._buffer = self._buffer[closed + len(CLOSE):]
                self.in_reasoning = False
                self._finished_reasoning = True
            else:
                opened = self._buffer.find(OPEN)
                if opened < 0:
                    hold = min(len(self._buffer), len(OPEN) - 1)
                    cut = len(self._buffer) - hold
                    visible.append(self._buffer[:cut])
                    self._buffer = self._buffer[cut:]
                    break
                visible.append(self._buffer[:opened])
                self._buffer = self._buffer[opened + len(OPEN):]
                self.in_reasoning = True
        return "".join(reasoning), "".join(visible)

    def flush(self) -> tuple[str, str]:
        tail, self._buffer = self._buffer, ""
        if not tail:
            return "", ""
        if self.in_reasoning:
            self.in_reasoning = False
            return tail, ""
        return "", tail
