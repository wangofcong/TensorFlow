from ast import main
from itertools import count
from re import I
from threading import Thread


import threading
import time
import multiprocessing
from concurrent.futures import ThreadPoolExecutor,as_completed

def write(content):
    with open(r"test.txt",'a',newline="") as file:
        file.write(content+"\n")
        file.write(content+"\n")
        file.write(content+"\n")
        file.write(content+"\n")
        file.write(content+"\n")

def single(contents):
    print("singel is start")
    for content in contents:
        write(content)
    print("single is end")

def multi(contents):
    print("multi is start")
    with ThreadPoolExecutor() as pool:
        futures=[pool.submit(write,content) for content in contents]
         
    print("multi is end")

def complete(lock:threading.Lock):

    global count
    count += 1
    print("this is ",count)


if __name__ == "__main__":
    count = 1
    contents = ["This is {} rows!".format(row) for row in range(100)]
    start = time.time()
    single(contents=contents)
    end = time.time()
    print("singele cost",end - start,"scores")
    start = time.time()
    multi(contents=contents)    
    end = time.time()
    print("multi cost",end - start,"scores!")

