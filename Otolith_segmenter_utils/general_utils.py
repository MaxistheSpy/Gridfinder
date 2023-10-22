from contextlib import contextmanager
import time
@contextmanager
def timer(name: str):
    track_time = True
    if not track_time:
        yield
        return
    print(name + "...")
    start_time = time.process_time()
    yield
    end_time = time.process_time()
    elapsed_time = end_time - start_time
    print(f"{name} took {time.strftime('%Hh%Mm%Ss', time.gmtime(elapsed_time))}")

def log_print(string):
    print(string)