import io

class AudioBuffer:
    """
    A circular-like buffer for accumulating audio chunks using bytearray.
    Optimized for 16kHz, Mono, Int16 PCM data.
    """
    def __init__(self):
        self.buffer = bytearray()

    def write(self, chunk: bytes):
        """Appends new audio bytes to the buffer."""
        self.buffer.extend(chunk)

    def read(self) -> bytes:
        """Reads the entire buffer and clears it."""
        data = bytes(self.buffer)
        self.buffer.clear()
        return data

    def clear(self):
        """Clears the buffer."""
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)
