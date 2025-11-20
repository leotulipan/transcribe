"""
Temporary patch for loguru to enable running without the actual loguru package.
This emulates the basic functionality of loguru.logger so transcribe_helpers can run.
"""

class MockLogger:
    def __init__(self):
        self.level = "INFO"

    def info(self, message):
        if self.level in ["INFO", "DEBUG"]:
            print(f"[INFO] {message}")

    def debug(self, message):
        if self.level == "DEBUG":
            print(f"[DEBUG] {message}")

    def warning(self, message):
        print(f"[WARNING] {message}")

    def error(self, message):
        print(f"[ERROR] {message}")

    def critical(self, message):
        print(f"[CRITICAL] {message}")

    def success(self, message):
        print(f"[SUCCESS] {message}")

    def exception(self, message):
        print(f"[EXCEPTION] {message}")

    def trace(self, message):
        if self.level == "TRACE":
            print(f"[TRACE] {message}")

    def level(self, name):
        self.level = name
        return self

# Create a mock logger instance to be used as a drop-in replacement
logger = MockLogger() 