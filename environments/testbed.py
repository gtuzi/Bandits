from abc import ABC, abstractmethod


class TestBed(ABC):
    @property
    @abstractmethod
    def best_mean(self) -> float:
        raise NotImplementedError

    @property
    @abstractmethod
    def best_arm(self) -> float:
        raise NotImplementedError