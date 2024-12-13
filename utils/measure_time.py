import time as t

def format_time(seconds):
    """Format hour:min:seconds"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02}:{minutes:02}:{seconds:05.2f}"

def measure_time(func):
    """Name says itself: to measure function time
    It prints: name, elapsed_time"""
    def wrapper(*args, **kwargs):
        start_time = t.time()
        result = func(*args, **kwargs)
        end_time = t.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time '{func.__name__}': {format_time(elapsed_time)}")
        return result
    return wrapper
