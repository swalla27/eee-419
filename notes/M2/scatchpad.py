import time

start = 10
counter = start

while counter > 0:
    print(f'Time remaining: {counter} seconds')
    counter -= 1
    time.sleep((1))

print("Time's up!")