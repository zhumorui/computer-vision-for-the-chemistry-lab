import time

def get_now_time():
    now_time = time.strftime("%Y-%m-%d-%H-%M-%S",time.localtime())

    return now_time

