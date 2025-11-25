class AudioBuffer:
    def __init__(self, window_size, step_size):
        self.window_size = window_size
        self.step_size = step_size
        self.buffer = bytearray()

    def write(self, chunk):
        self.buffer.extend(chunk)

    def is_ready(self):
        return len(self.buffer) >= self.window_size

    def get_window(self):
        # Return current window
        window = self.buffer[:self.window_size]
        # Slide buffer
        self.buffer = self.buffer[self.step_size:]
        return window

    def clear(self):
        self.buffer = bytearray()
