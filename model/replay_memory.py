from collections import namedtuple
import random

class ReplayMemory(object):
    def __init__(self,capacity=1000000):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.Transition = namedtuple('Transition', ('state', 'action', 'next_state','reward'))
    
    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        
        self.memory[self.position] = self.Transition(*args)

    def sample(self, batch_size):
        assert batch_size <= len(self.memory)

        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)