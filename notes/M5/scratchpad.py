import numpy as np

rng = np.random.default_rng()
x = ['cat', 'dog', 'fish']
print(rng.choice(x, 10))


class DogBreeds():
    def __init__(self, type, name, sound):
        self.type = type
        self.name = name
        self.sound = sound
    
    def bark(self):
        print(self.sound)


