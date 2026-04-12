from abc import ABC, abstractmethod

class Action:
    pass

class Observation:
    pass

class State:
    pass

class Environment(ABC):

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action):
        pass

    @property
    def state(self):
        pass