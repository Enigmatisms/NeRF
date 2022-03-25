#-*-coding:utf-8-*-
"""
    Running average timer
    @author Enigmatisms @date 2022.3.25
    @copyright 2022
"""
from time import time
from collections import deque
from datetime import timedelta

class Timer:
    def __init__(self, max_len) -> None:
        self.deque = deque(maxlen = max_len)
        self.last_time = 0.

    def get_mean_time(self):
        return sum(self.deque) / len(self.deque)
    
    def tic(self):
        self.last_time = time()

    def toc(self):
        self.deque.append(time() - self.last_time)
        return self.get_mean_time()

    def remaining_time(self, exec_needed:int):
        time = self.get_mean_time() * exec_needed
        return str(timedelta(seconds = time))