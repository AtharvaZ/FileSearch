"""
Shared exceptions and signal handlers for FileSearch
"""
import signal

class TimeoutException(Exception):
    """Raised when a file operation times out"""
    pass

def timeout_handler(signum, frame):
    """Signal handler for SIGALRM timeout"""
    raise TimeoutException()
