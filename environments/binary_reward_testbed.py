from abc import ABC, abstractmethod

import numpy as np
from typing import List, Optional
from functools import partial
from tools.random_walks import BernoulliRandomWalk, RandomWalk, NormalRandomWalk


class Bandit(ABC):
    def __init__(self):
        self._call_count = 0
        self._rand = NotImplementedError

    @property
    def success_rate(self) -> float:
        raise NotImplementedError

    @property
    def call_count(self):
        return self._call_count

    @abstractmethod
    def sample(self, n_samples: int) -> List[float]:
        assert n_samples > 0
        raise NotImplementedError

    def __call__(self, n_samples: int = 1) -> List[float]:
        self._call_count += 1
        return self.sample(n_samples)


class StationaryBrnoulliBandit(Bandit):
    def __init__(self, success_rate: float):
        assert 0 <= success_rate <= 1, 'Invalid success rate'
        super(StationaryBrnoulliBandit, self).__init__()
        self._p = success_rate
        self.rand = partial(np.random.binomial, n=1)

    @property
    def success_rate(self) -> float:
        return self._p

    def sample(self, n_samples: int) -> List[float]:
        return self.rand(p=self.success_rate, size=n_samples).astype(dtype=np.float).tolist()


class NonStationaryBernoulliBandit(Bandit):
    def __init__(self, initial_success_rate: float, random_walk: str = 'normal', randomness_scale: float = 0.1):
        assert 0 <= initial_success_rate <= 1, 'Invalid success rate'
        super(NonStationaryBernoulliBandit, self).__init__()
        if random_walk.lower() == 'normal':
            self.rw: RandomWalk = NormalRandomWalk(
                initial_value=initial_success_rate,
                sig=randomness_scale,
                clip_min=0,
                clip_max=1.)

        elif random_walk.lower() == 'bernoulli':
            self.rw: RandomWalk = BernoulliRandomWalk(
                initial_value=initial_success_rate,
                p=randomness_scale,
                clip_min=0,
                clip_max=1.)
        else:
            raise Exception('Distribution not recognized')

    def sample(self, n_samples: int) -> List[float]:
        return self.rw(size=n_samples).tolist()

    def success_rate(self) -> float:
        return float(self.rw.current_mean)


class TestBed:
    def __init__(
            self,
            success_rates: List,
            reward_randomness_scales: List = [],
            random_walk='normal',
            stationary: bool = True):

        if stationary:
            self.bandits = [StationaryBrnoulliBandit(success_rate=p) for p in success_rates]
        else:
            assert len(success_rates) == len(reward_randomness_scales)
            self.bandits = [
                NonStationaryBernoulliBandit(
                    initial_success_rate=p,
                    randomness_scale=r_scale,
                    random_walk=random_walk)
                for p, r_scale in zip(success_rates, reward_randomness_scales)]

    def __call__(self, action: int) -> float:
        return self.bandits[action](n_samples=1)[0]

    @property
    def best_success_rate(self) -> float:
        return max([b.success_rate() for b in self.bandits])

    @property
    def best_arm(self) -> float:
        success_rates = [b.success_rate() for b in self.bandits]
        return success_rates.index(max(success_rates))


if __name__ == '__main__':
    p = 0.7
    N = 100
    bandit = StationaryBrnoulliBandit(p)
    samples = [bandit()[0] for _ in range(N)]
    print('p, {}, p_hat: {:.4f}: '.format(p, float(np.mean(samples))))

    exit(0)