from abc import ABC, abstractmethod


class AgentBase(ABC):

    @abstractmethod
    def before(self, *args, **kwargs):
        pass

    @abstractmethod
    def after(self, *args, **kwargs):
        pass

    @abstractmethod
    def act(self, state) -> int:
        pass

    @abstractmethod
    def learn(self, *args, **kwargs):
        pass
