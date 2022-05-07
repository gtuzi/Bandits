import os.path
from typing import Callable, List, Tuple, Dict
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from environments.testbed import NonAssocativeTestBed
from utils import mk_clear_dir

from environments.continuous_reward_testbed import ContinuousValueRewardTestBed
from environments.binary_reward_testbed import BinaryValueRewardTestBed

from tools.coefficients import (
    ConstantCoefficient,
    CosineDecayCoefficient)

from value_functions.nonassocative_value_functions import (
    QMonteCarlo,
    QCoefficientMovingAverage)

from policies.nonassociative_policies import (
    Policy,
    EpsGreedyPolicy,
    UCB1Policy,
    NaiivePreferencePolicy,
    SoftmaxExplorationPolicy,
    BernoulliPolicy,
    BernoulliGreedy,
    BernoulliThompsonSampling)


def get_cont_reward_test_bed(reward_means, reward_randomness_scale, stationary=False):
    return ContinuousValueRewardTestBed(
        reward_means=reward_means,
        reward_randomness_scales=reward_randomness_scale,
        stationary=stationary)


def get_binary_reward_test_bed(
        success_rates: List,
        reward_randomness_scales: List = [],
        stationary: bool = True):
    return BinaryValueRewardTestBed(
        success_rates=success_rates,
        reward_randomness_scales=reward_randomness_scales,
        stationary=stationary)


def simulate_epsilon_greedy(
        test_bed_constructor: Callable[[], NonAssocativeTestBed],
        policy_constructor: Callable[[float, ], EpsGreedyPolicy],
        epsilons: List[float],
        n_trials: int,
        n_steps: int,
        desc='') -> Tuple[Dict, Dict, Dict, Dict]:
    rewards = {str(eps): [list() for _ in range(n_trials)] for eps in epsilons}
    cummulative_best_means = {str(eps): [[0.] for _ in range(n_trials)] for eps in epsilons}
    cummulative_rewards = {str(eps): [[0.] for _ in range(n_trials)] for eps in epsilons}
    best_arms = {str(eps): [list() for _ in range(n_trials)] for eps in epsilons}

    for eps in epsilons:
        for trial in tqdm(range(n_trials), desc=f'{desc} - epsilon: {eps} - trials'):
            policy = policy_constructor(eps)
            test_bed = test_bed_constructor()
            for step in range(1, n_steps + 1):
                a = policy(step=step)
                r = test_bed(action=a)
                policy.step(step=step, action=a, reward=r)
                rewards[str(eps)][trial].append(r)
                cummulative_best_means[str(eps)][trial].append(
                    test_bed.best_mean + cummulative_best_means[str(eps)][trial][-1])
                cummulative_rewards[str(eps)][trial].append(r + cummulative_rewards[str(eps)][trial][-1])
                best_arms[str(eps)][trial] = test_bed.best_arm

    return rewards, cummulative_rewards, cummulative_best_means, best_arms


def simulate_ucb(
        test_bed_constructor: Callable[[], NonAssocativeTestBed],
        policy_constructor: Callable[[float, ], UCB1Policy],
        uncertainty_coefficients: List[float],
        n_trials: int,
        n_steps: int,
        desc='') -> Tuple[Dict, Dict, Dict, Dict]:
    rewards = {str(c): [list() for _ in range(n_trials)] for c in uncertainty_coefficients}
    cummulative_best_means = {str(c): [[0.] for _ in range(n_trials)] for c in uncertainty_coefficients}
    cummulative_rewards = {str(c): [[0.] for _ in range(n_trials)] for c in uncertainty_coefficients}
    best_arms = {str(c): [list() for _ in range(n_trials)] for c in uncertainty_coefficients}

    for c in uncertainty_coefficients:
        for trial in tqdm(range(n_trials), desc=f'{desc} - uncertainty coefficient: {c} - trials'):
            policy = policy_constructor(c)
            test_bed = test_bed_constructor()
            for step in range(1, n_steps + 1):
                a = policy(step=step)
                r = test_bed(action=a)
                policy.step(step=step, action=a, reward=r)
                rewards[str(c)][trial].append(r)
                cummulative_best_means[str(c)][trial].append(
                    test_bed.best_mean + cummulative_best_means[str(c)][trial][-1])
                cummulative_rewards[str(c)][trial].append(r + cummulative_rewards[str(c)][trial][-1])
                best_arms[str(c)][trial] = test_bed.best_arm

    return rewards, cummulative_rewards, cummulative_best_means, best_arms


def simulate_bernoulli_testbed(
        test_bed_constructor: Callable[[], NonAssocativeTestBed],
        policy_constructor: Callable[[], BernoulliPolicy],):

    sentinel = 'none'
    rewards = {sentinel: [list() for _ in range(n_trials)]}
    cummulative_best_means = {sentinel: [[0.] for _ in range(n_trials)]}
    cummulative_rewards = {sentinel: [[0.] for _ in range(n_trials)]}
    best_arms = {sentinel: [list() for _ in range(n_trials)]}

    for trial in tqdm(range(n_trials), desc=f'Bernoulli TestBed'):
        policy = policy_constructor()
        test_bed = test_bed_constructor()
        for step in range(1, n_steps + 1):
            a = policy(step=step)
            r = test_bed(action=a)
            policy.step(step=step, action=a, reward=r)
            rewards[sentinel][trial].append(r)
            cummulative_best_means[sentinel][trial].append(
                test_bed.best_mean + cummulative_best_means[sentinel][trial][-1])
            cummulative_rewards[sentinel][trial].append(r + cummulative_rewards[sentinel][trial][-1])
            best_arms[sentinel][trial] = test_bed.best_arm

    return rewards, cummulative_rewards, cummulative_best_means, best_arms


def simulate_paramterized_policy(
        test_bed_constructor: Callable[[], NonAssocativeTestBed],
        policy_constructor: Callable[[float, ], Policy],
        varying_parameters: List[float],
        n_trials: int,
        n_steps: int,
        desc='') -> Tuple[Dict, Dict, Dict, Dict]:
    rewards = {str(c): [list() for _ in range(n_trials)] for c in varying_parameters}
    cummulative_best_means = {str(c): [[0.] for _ in range(n_trials)] for c in varying_parameters}
    cummulative_rewards = {str(c): [[0.] for _ in range(n_trials)] for c in varying_parameters}
    best_arms = {str(c): [list() for _ in range(n_trials)] for c in varying_parameters}

    for param in varying_parameters:
        for trial in tqdm(range(n_trials), desc=f'{desc} parameter: {param} - trials'):
            policy = policy_constructor(param)
            test_bed = test_bed_constructor()
            for step in range(1, n_steps + 1):
                a = policy(step=step)
                r = test_bed(action=a)
                policy.step(step=step, action=a, reward=r)
                rewards[str(param)][trial].append(r)
                cummulative_best_means[str(param)][trial].append(
                    test_bed.best_mean + cummulative_best_means[str(param)][trial][-1])
                cummulative_rewards[str(param)][trial].append(r + cummulative_rewards[str(param)][trial][-1])
                best_arms[str(param)][trial] = test_bed.best_arm

    return rewards, cummulative_rewards, cummulative_best_means, best_arms



######## Continuous valued rewards ###############

def experiment_1(n_steps, n_trials):
    """
        For a fixed action-value (E[R | a]) for each bandit, compare the performance of epsilon greedy
        policies accross different values of exploration.

        Q uses sample averaging (unbiased estimator)
    """

    n_bandits = 10  # Each bandit is triggered by one action
    Q0 = 0.0
    epsilons = [0.0, 0.01, 0.1, 0.3]
    reward_randomness_scale = 0.00
    plot_root_name = 'experiment_1'

    test_bed_constructor = lambda: get_cont_reward_test_bed(
        reward_means=np.random.normal(0., 1., size=n_bandits).tolist(),
        reward_randomness_scale=[reward_randomness_scale] * n_bandits,
        stationary=True)

    q_constructor = lambda: QMonteCarlo(n_actions=n_bandits, initial_action_value=Q0)
    policy_constructor = lambda _e: EpsGreedyPolicy(q=q_constructor(), eps=_e)

    rewards, cummulative_rewards, cummulative_best_means, best_arms = simulate_epsilon_greedy(
        test_bed_constructor=test_bed_constructor,
        policy_constructor=policy_constructor,
        epsilons=epsilons,
        n_trials=n_trials,
        n_steps=n_steps,
        desc='Experiment 1')

    reward_averages = dict()
    regret_averages = dict()

    for eps in epsilons:
        reward_averages[str(eps)] = [
            np.mean([rewards[str(eps)][trial][step] for trial in range(n_trials)]) for step in range(n_steps)]

        regret_averages[str(eps)] = [
            np.mean([(cummulative_best_means[str(eps)][trial][step] - cummulative_rewards[str(eps)][trial][step]) / step
                     for trial in range(n_trials)])
            for step in range(1, n_steps + 1)]

    d = os.path.join(os.getcwd(), 'plots')
    _ = mk_clear_dir(d, False)

    _ = plt.figure()
    for eps in epsilons:
        plt.plot(reward_averages[str(eps)])
    plt.legend(['eps: ' + str(eps) for eps in epsilons])
    plt.ylabel('Average Reward')
    plt.xlabel('Simulation Step')
    plt.title('Experiment 1: Eps-Greedy\nAvg. Rewards')
    plt.grid()
    try:
        plt.savefig(os.path.join(d, f'rewards_{plot_root_name}.png'))
    except:
        print(f'Could not save rewards_{plot_root_name} plot')
    finally:
        plt.show()

    _ = plt.figure()
    for eps in epsilons:
        plt.plot(regret_averages[str(eps)])
    plt.legend(['eps: ' + str(eps) for eps in epsilons])
    plt.ylabel('Average Regret')
    plt.xlabel('Simulation Step')
    plt.title('Experiment 1: Eps-Greedy\nAvg. Regrets')
    plt.grid()
    try:
        plt.savefig(os.path.join(d, f'regrets_{plot_root_name}.png'))
    except:
        print(f'Could not save regrets_{plot_root_name} plot')
    finally:
        plt.show()


def experiment_2(n_steps, n_trials):
    """
        Compare the performances across different levels of reward noise
    """

    n_bandits = 10  # Each bandit is triggered by one action
    Q0 = 0.0
    epsilon = 0.1
    reward_randomness_scales = [0.01, 0.1, 1.0, 3.0]
    rewards_avgs_collector = dict()
    regret_avgs_collector = dict()
    plot_root_name = 'experiment_2'

    for reward_randomness_scale in reward_randomness_scales:
        test_bed_constructor = lambda: get_cont_reward_test_bed(
            reward_means=np.random.normal(0., 1., size=n_bandits).tolist(),
            reward_randomness_scale=[reward_randomness_scale] * n_bandits,
            stationary=True)

        q_constructor = lambda: QMonteCarlo(n_actions=n_bandits, initial_action_value=Q0)
        policy_constructor = lambda _e: EpsGreedyPolicy(q=q_constructor(), eps=_e)

        print('Reward randomness scale: ', reward_randomness_scale)
        rewards, cummulative_rewards, cummulative_best_means, best_arms = simulate_epsilon_greedy(
            test_bed_constructor=test_bed_constructor,
            policy_constructor=policy_constructor,
            epsilons=[epsilon],
            n_trials=n_trials,
            n_steps=n_steps,
            desc='Experiment 2')

        reward_averages = [np.mean([rewards[str(epsilon)][trial][step] for trial in range(n_trials)]) for step in
                           range(n_steps)]
        rewards_avgs_collector[reward_randomness_scale] = reward_averages

        regret_averages = [
            np.mean([cummulative_best_means[str(epsilon)][trial][step] - cummulative_rewards[str(epsilon)][trial][step]
                     for trial in range(n_trials)])
            for step in range(1, n_steps + 1)]

        regret_avgs_collector[reward_randomness_scale] = regret_averages

    d = os.path.join(os.getcwd(), 'plots')
    _ = mk_clear_dir(d, False)

    _ = plt.figure()
    for reward_randomness_scale in reward_randomness_scales:
        plt.plot(rewards_avgs_collector[reward_randomness_scale])
    plt.legend(['Randomness scale : ' + str(reward_randomness_scale)
                for reward_randomness_scale in reward_randomness_scales])
    plt.ylabel('Average Reward')
    plt.xlabel('Simulation Step')
    plt.title('Experiment 2\nAverage Rewards')
    plt.grid()
    try:
        plt.savefig(os.path.join(d, f'rewards_{plot_root_name}.png'))
    except:
        print(f'Could not save {plot_root_name} plot')
    finally:
        plt.show()

    _ = plt.figure()
    for reward_randomness_scale in reward_randomness_scales:
        plt.plot(regret_avgs_collector[reward_randomness_scale])
    plt.legend(['Randomness scale : ' + str(reward_randomness_scale)
                for reward_randomness_scale in reward_randomness_scales])
    plt.ylabel('Avg. Regret')
    plt.xlabel('Simulation Step')
    plt.title('Experiment 2\nAverage Regrets')
    plt.grid()
    try:
        plt.savefig(os.path.join(d, f'regrets_{plot_root_name}.png'))
    except:
        print(f'Could not save regrets {plot_root_name} plot')
    finally:
        plt.show()


def experiment_3(n_steps, n_trials):
    """
            Compare the performances between the sample average (MC) Q and constant stepsize Q
            on a **stationary** testbed
        """

    n_bandits = 10  # Each bandit is triggered by one action
    Q0 = 0.0
    epsilons = [0., 0.01, 0.1, 0.2]
    alpha = 0.1
    reward_randomness_scale = 1.0
    plot_root_name = 'experiment_3'

    test_bed_constructor = lambda: get_cont_reward_test_bed(
        reward_means=np.random.normal(0., 1., size=n_bandits).tolist(),
        reward_randomness_scale=[reward_randomness_scale] * n_bandits,
        stationary=True)

    q_constructor = lambda: QMonteCarlo(n_actions=n_bandits, initial_action_value=Q0)
    policy_constructor = lambda _e: EpsGreedyPolicy(q=q_constructor(), eps=_e)

    # Monte carlo
    rewards_mc, cummulative_rewards_mc, cummulative_best_means_mc, best_arms_mc = simulate_epsilon_greedy(
        test_bed_constructor=test_bed_constructor,
        policy_constructor=policy_constructor,
        epsilons=epsilons,
        n_trials=n_trials,
        n_steps=n_steps,
        desc='Experiment 3')

    q_constructor = lambda: QCoefficientMovingAverage(n_actions=n_bandits, coefficient=ConstantCoefficient(alpha),
                                                      initial_action_value=Q0)
    policy_constructor = lambda _e: EpsGreedyPolicy(q=q_constructor(), eps=_e)

    # Constant coefficient
    rewards_cc, cummulative_rewards_cc, cummulative_best_means_cc, best_arms_cc = simulate_epsilon_greedy(
        test_bed_constructor=test_bed_constructor,
        policy_constructor=policy_constructor,
        epsilons=epsilons,
        n_trials=n_trials,
        n_steps=n_steps,
        desc='Experiment 3')

    reward_avgs_mc = dict()
    regret_avgs_mc = dict()
    for eps in epsilons:
        reward_avgs_mc[str(eps)] = [
            np.mean([rewards_mc[str(eps)][trial][step] for trial in range(n_trials)]) for step in range(n_steps)]
        regret_avgs_mc[str(eps)] = [
            np.mean([cummulative_best_means_mc[str(eps)][trial][step] - cummulative_rewards_mc[str(eps)][trial][step]
                     for trial in range(n_trials)])
            for step in range(1, n_steps + 1)]

    reward_avgs_cc = dict()
    regret_avgs_cc = dict()
    for eps in epsilons:
        reward_avgs_cc[str(eps)] = [
            np.mean([rewards_cc[str(eps)][trial][step] for trial in range(n_trials)]) for step in range(n_steps)]
        regret_avgs_cc[str(eps)] = [
            np.mean([cummulative_best_means_cc[str(eps)][trial][step] - cummulative_rewards_cc[str(eps)][trial][step]
                     for trial in range(n_trials)])
            for step in range(1, n_steps + 1)]

    d = os.path.join(os.getcwd(), 'plots')
    _ = mk_clear_dir(d, False)

    _ = plt.figure()
    labels = []
    for eps in epsilons:
        plt.plot(reward_avgs_mc[str(eps)])
        labels.append('MC - eps: ' + str(eps))
        plt.plot(reward_avgs_cc[str(eps)])
        labels.append('CC - eps: ' + str(eps))
    plt.legend(labels)
    plt.ylabel('Avg. Reward')
    plt.xlabel('Simulation Step')
    plt.title('Experiment 3: MC-Q vs Constant-Coefficient-Q\nStationary Evironment')
    plt.grid()
    try:
        plt.savefig(os.path.join(d, f'rewards_{plot_root_name}.png'))
    except:
        print(f'Could not save {plot_root_name} plot')
    finally:
        plt.show()

    _ = plt.figure()
    labels = []
    for eps in epsilons:
        plt.plot(regret_avgs_mc[str(eps)])
        labels.append('MC - eps: ' + str(eps))
        plt.plot(regret_avgs_cc[str(eps)])
        labels.append('CC - eps: ' + str(eps))
    plt.legend(labels)
    plt.ylabel('Avg. Regret')
    plt.xlabel('Simulation Step')
    plt.title('Experiment 3: MC-Q vs Constant-Coefficient-Q\nRegrets')
    plt.grid()
    try:
        plt.savefig(os.path.join(d, f'regrets_{plot_root_name}.png'))
    except:
        print(f'Could not save regrets {plot_root_name} plot')
    finally:
        plt.show()


def experiment_4(n_steps, n_trials):
    """
        Compare the performances between the sample average (MC) Q and constant stepsize Q
        on a **non-stationary** testbed
    """

    n_bandits = 10  # Each bandit is triggered by one action
    Q0 = 0.0
    epsilons = [0., 0.01, 0.1, 0.2]
    alpha = 0.1
    reward_randomness_scale = 1.0
    plot_root_text = 'experiment_4'

    test_bed_constructor = lambda: get_cont_reward_test_bed(
        reward_means=np.random.normal(0., 1., size=n_bandits).tolist(),
        reward_randomness_scale=[reward_randomness_scale] * n_bandits,
        stationary=False)

    q_constructor = lambda: QMonteCarlo(n_actions=n_bandits, initial_action_value=Q0)
    policy_constructor = lambda _e: EpsGreedyPolicy(q=q_constructor(), eps=_e)

    # Monte carlo
    rewards_mc, cummulative_rewards_mc, cummulative_best_means_mc, best_arms_mc = simulate_epsilon_greedy(
        test_bed_constructor=test_bed_constructor,
        policy_constructor=policy_constructor,
        epsilons=epsilons,
        n_trials=n_trials,
        n_steps=n_steps,
        desc='Experiment 4 - QMC')

    q_constructor = lambda: QCoefficientMovingAverage(
        n_actions=n_bandits, coefficient=ConstantCoefficient(alpha), initial_action_value=Q0)
    policy_constructor = lambda _e: EpsGreedyPolicy(q=q_constructor(), eps=_e)

    # Constant coefficient
    rewards_cc, cummulative_rewards_cc, cummulative_best_means_cc, best_arms_cc = simulate_epsilon_greedy(
        test_bed_constructor=test_bed_constructor,
        policy_constructor=policy_constructor,
        epsilons=epsilons,
        n_trials=n_trials,
        n_steps=n_steps,
        desc='Experiment 4 - QCC')

    reward_avgs_mc = dict()
    regret_avgs_mc = dict()
    for eps in epsilons:
        reward_avgs_mc[str(eps)] = [
            np.mean([rewards_mc[str(eps)][trial][step] for trial in range(n_trials)]) for step in range(n_steps)]
        regret_avgs_mc[str(eps)] = [
            np.mean([cummulative_best_means_mc[str(eps)][trial][step] - cummulative_rewards_mc[str(eps)][trial][step]
                     for trial in range(n_trials)])
            for step in range(1, n_steps + 1)]

    reward_avgs_cc = dict()
    regret_avgs_cc = dict()
    for eps in epsilons:
        reward_avgs_cc[str(eps)] = [
            np.mean([rewards_cc[str(eps)][trial][step] for trial in range(n_trials)]) for step in range(n_steps)]
        regret_avgs_cc[str(eps)] = [
            np.mean([cummulative_best_means_cc[str(eps)][trial][step] - cummulative_rewards_cc[str(eps)][trial][step]
                     for trial in range(n_trials)])
            for step in range(1, n_steps + 1)]

    d = os.path.join(os.getcwd(), 'plots')
    _ = mk_clear_dir(d, False)

    _ = plt.figure()
    labels = []
    for eps in epsilons:
        plt.plot(reward_avgs_mc[str(eps)])
        labels.append('MC - eps: ' + str(eps))
        plt.plot(reward_avgs_cc[str(eps)])
        labels.append('CC - eps: ' + str(eps))
    plt.legend(labels)
    plt.ylabel('Average Reward')
    plt.xlabel('Simulation Step')
    plt.grid()
    plt.title('Experiment 4: MC-Q vs Constant-Coefficient-Q\nNon-stationary Environment')
    try:
        plt.savefig(os.path.join(d, f'rewards_{plot_root_text}.png'))
    except:
        print(f'Could not save rewards_{plot_root_text} plot')
    finally:
        plt.show()

    _ = plt.figure()
    labels = []
    for eps in epsilons:
        plt.plot(regret_avgs_mc[str(eps)])
        labels.append('MC - eps: ' + str(eps))
        plt.plot(regret_avgs_cc[str(eps)])
        labels.append('CC - eps: ' + str(eps))
    plt.legend(labels)
    plt.ylabel('Average Regret')
    plt.xlabel('Simulation Step')
    plt.grid()
    plt.title('Experiment 4: MC-Q vs Constant-Coefficient-Q\nNon-stationary Environment')
    try:
        plt.savefig(os.path.join(d, f'regrets_{plot_root_text}.png'))
    except:
        print(f'Could not save regret_{plot_root_text} plot')
    finally:
        plt.show()


def experiment_5(n_steps, n_trials):
    """
        Compare pessimistic vs optimistic initial value function (Q)
    """

    n_bandits = 10  # Each bandit is triggered by one action
    Q0_pessimistic = 0.0
    Q0_optimistic = 10.0
    epsilons = [0., 0.1]
    reward_randomness_scale = 1.0
    plot_root_text = 'experiment_5'

    test_bed_constructor = lambda: get_cont_reward_test_bed(
        reward_means=np.random.normal(0., 1., size=n_bandits).tolist(),
        reward_randomness_scale=[reward_randomness_scale] * n_bandits,
        stationary=True)

    q_constructor = lambda: QMonteCarlo(n_actions=n_bandits, initial_action_value=Q0_pessimistic)
    policy_constructor = lambda _e: EpsGreedyPolicy(q=q_constructor(), eps=_e)

    rewards_pessimistic, cummulative_rewards_pessimistic, cummulative_best_means_pessimistic, best_arms_pessimistic = \
        simulate_epsilon_greedy(
            test_bed_constructor=test_bed_constructor,
            policy_constructor=policy_constructor,
            epsilons=epsilons,
            n_trials=n_trials,
            n_steps=n_steps,
            desc='Experiment 5 - Pessimistic')

    q_constructor = lambda: QMonteCarlo(n_actions=n_bandits, initial_action_value=Q0_optimistic)
    policy_constructor = lambda _e: EpsGreedyPolicy(q=q_constructor(), eps=_e)

    rewards_optimisitic, cummulative_rewards_optimisitic, cummulative_best_means_optimisitic, best_arms_optimisitic = \
        simulate_epsilon_greedy(
            test_bed_constructor=test_bed_constructor,
            policy_constructor=policy_constructor,
            epsilons=epsilons,
            n_trials=n_trials,
            n_steps=n_steps,
            desc='Experiment 5 - Optimistic')

    rewards_avgs_pessimistic = dict()
    rewards_avgs_optimistic = dict()
    regrets_avgs_pessimistic = dict()
    regrets_avgs_optimisitic = dict()

    for epsilon in epsilons:
        rewards_avgs_pessimistic[str(epsilon)] = [
            np.mean([rewards_pessimistic[str(epsilon)][trial][step] for trial in range(n_trials)])
            for step in range(n_steps)]

        rewards_avgs_optimistic[str(epsilon)] = [
            np.mean([rewards_optimisitic[str(epsilon)][trial][step] for trial in range(n_trials)])
            for step in range(n_steps)]

        regrets_avgs_pessimistic[str(epsilon)] = [
            np.mean([cummulative_best_means_pessimistic[str(epsilon)][trial][step] -
                     cummulative_rewards_pessimistic[str(epsilon)][trial][step]
                     for trial in range(n_trials)])
            for step in range(1, n_steps + 1)]

        regrets_avgs_optimisitic[str(epsilon)] = [
            np.mean([cummulative_best_means_optimisitic[str(epsilon)][trial][step] -
                     cummulative_rewards_optimisitic[str(epsilon)][trial][step]
                     for trial in range(n_trials)])
            for step in range(1, n_steps + 1)]

    d = os.path.join(os.getcwd(), 'plots')
    _ = mk_clear_dir(d, False)

    _ = plt.figure()
    legend = []
    for epsilon in epsilons:
        plt.plot(rewards_avgs_pessimistic[str(epsilon)])
        legend.append(f'Pessimistic: eps: {epsilon}, Q0: {Q0_pessimistic}')
        plt.plot(rewards_avgs_optimistic[str(epsilon)])
        legend.append(f'Optimisitic: eps: {epsilon}, Q0: {Q0_optimistic}')
    plt.legend(legend)
    plt.ylabel('Average Reward')
    plt.xlabel('Simulation Step')
    plt.grid()
    plt.title('Experiment 5: Q(0) value comparisons\nReward')
    try:
        plt.savefig(os.path.join(d, f'rewards_{plot_root_text}.png'))
    except:
        print(f'Could not save rewards_{plot_root_text} plots')
    finally:
        plt.show()

    _ = plt.figure()
    legend = []
    for epsilon in epsilons:
        plt.plot(regrets_avgs_pessimistic[str(epsilon)])
        legend.append(f'Pessimistic: eps: {epsilon}, Q0: {Q0_pessimistic}')
        plt.plot(regrets_avgs_optimisitic[str(epsilon)])
        legend.append(f'Optimisitic: eps: {epsilon}, Q0: {Q0_optimistic}')
    plt.legend(legend)
    plt.ylabel('Average Regret')
    plt.xlabel('Simulation Step')
    plt.title('Experiment 5: Q0 value comparisons\nRegret')
    try:
        plt.savefig(os.path.join(d, f'regrets_{plot_root_text}.png'))
    except:
        print(f'Could not save regrets_{plot_root_text} plots')
    finally:
        plt.show()


def experiment_6(n_steps, n_trials):
    """
        Compare UCB policy with eps-greedy on a stationary test bed
    :return:
    """

    n_bandits = 10  # Each bandit is triggered by one action
    Q0 = 0.0
    epsilon = 0.1  # Same as in Sutton book (Sec. 2.7)
    c = 2  # Same as in Sutton book (Sec. 2.7)
    reward_randomness_scale = 1.0
    plot_root_text = 'experiment_6'

    test_bed_constructor = lambda: get_cont_reward_test_bed(
        reward_means=np.random.normal(0., 1., size=n_bandits).tolist(),
        reward_randomness_scale=[reward_randomness_scale] * n_bandits,
        stationary=True)

    q_constructor = lambda: QMonteCarlo(n_actions=n_bandits, initial_action_value=Q0)
    policy_constructor = lambda _e: EpsGreedyPolicy(q=q_constructor(), eps=_e)

    rewards_eps_greedy, cummulative_rewards_eps_greedy, cummulative_best_means_eps_greedy, best_arms_eps_greedy = \
        simulate_epsilon_greedy(
            test_bed_constructor=test_bed_constructor,
            policy_constructor=policy_constructor,
            epsilons=[epsilon],
            n_trials=n_trials,
            n_steps=n_steps,
            desc='Experiment 6 - Eps-Greedy')

    rewards_avgs_eps_greedy = [
        np.mean([rewards_eps_greedy[str(epsilon)][trial][step] for trial in range(n_trials)])
        for step in range(n_steps)]

    regrets_avgs_eps_greedy = [
        np.mean([cummulative_best_means_eps_greedy[str(epsilon)][trial][step] -
                 cummulative_rewards_eps_greedy[str(epsilon)][trial][step]
                 for trial in range(n_trials)])
        for step in range(1, n_steps + 1)]

    q_constructor = lambda: QMonteCarlo(n_actions=n_bandits, initial_action_value=Q0)
    policy_constructor = lambda _c: UCB1Policy(q=q_constructor(), c=_c)

    rewards_ucb, cummulative_rewards_ucb, cummulative_best_means_ucb, best_arms_ucb = simulate_ucb(
        test_bed_constructor=test_bed_constructor,
        policy_constructor=policy_constructor,
        uncertainty_coefficients=[c],
        n_trials=n_trials,
        n_steps=n_steps,
        desc='Experiment 6 - UCB')

    rewards_avgs_ucb = [
        np.mean([rewards_ucb[str(c)][trial][step] for trial in range(n_trials)])
        for step in range(n_steps)]

    regrets_avgs_ucb = [
        np.mean([cummulative_best_means_ucb[str(c)][trial][step] -
                 cummulative_rewards_ucb[str(c)][trial][step]
                 for trial in range(n_trials)])
        for step in range(1, n_steps + 1)]

    d = os.path.join(os.getcwd(), 'plots')
    _ = mk_clear_dir(d, False)

    _ = plt.figure()
    plt.plot(rewards_avgs_eps_greedy)
    plt.plot(rewards_avgs_ucb)
    plt.legend([f'Eps-Greedy: eps {epsilon}', f'UCB: c {c}'])
    plt.ylabel('Average Reward')
    plt.xlabel('Simulation Step')
    plt.title('Experiment 6: UCB vs EpsGreedy')
    plt.grid()
    try:
        plt.savefig(os.path.join(d, f'rewards_{plot_root_text}.png'))
    except:
        print(f'Could not save rewards_{plot_root_text} plots')
    finally:
        plt.show()

    plt.plot(regrets_avgs_eps_greedy)
    plt.plot(regrets_avgs_ucb)
    plt.legend([f'Eps-Greedy: eps {epsilon}', f'UCB: c {c}'])
    plt.ylabel('Average Regret')
    plt.xlabel('Simulation Step')
    plt.title('Experiment 6: UCB vs EpsGreedy')
    plt.grid()
    try:
        plt.savefig(os.path.join(d, f'regrets_{plot_root_text}.png'))
    except:
        print(f'Could not save regrets_{plot_root_text} plots')
    finally:
        plt.show()


def experiment_7(n_steps, n_trials):
    """
            Naiive preference policy over different constant temperatures
    :return:
    """
    n_bandits = 10  # Each bandit is triggered by one action
    H0 = 0.0
    Rbar0 = 0.0
    alpha = 0.1
    reward_randomness_scale = 1.0
    constant_temperatures = [0.1, 0.5, 1.0, 2.0, 4.0]
    plot_root_name = 'experiment_7'

    test_bed_constructor = lambda: get_cont_reward_test_bed(
        reward_means=np.random.normal(0., 1., size=n_bandits).tolist(),
        reward_randomness_scale=[reward_randomness_scale] * n_bandits,
        stationary=True)

    policy_constructor = lambda _temp: NaiivePreferencePolicy(
        n_actions=n_bandits,
        preference_initial_value=H0,
        reward_initial_value=Rbar0,
        learning_rate=alpha,
        temperature=ConstantCoefficient(_temp)
    )

    rewards, cummulative_rewards, cummulative_best_means, best_arms = simulate_paramterized_policy(
        test_bed_constructor=test_bed_constructor,
        policy_constructor=policy_constructor,
        varying_parameters=constant_temperatures,
        n_trials=n_trials,
        n_steps=n_steps,
        desc='Experiment 7, temperature')

    rewards_avgs = dict()
    regret_averages = dict()
    for tmp in constant_temperatures:
        rewards_avgs[str(tmp)] = [
            np.mean([rewards[str(tmp)][trial][step] for trial in range(n_trials)]) for step in range(n_steps)]

        regret_averages[str(tmp)] = [
            np.mean([cummulative_best_means[str(tmp)][trial][step] - cummulative_rewards[str(tmp)][trial][step]
                     for trial in range(n_trials)])
            for step in range(1, n_steps + 1)]

    d = os.path.join(os.getcwd(), 'plots')
    _ = mk_clear_dir(d, False)

    _ = plt.figure()
    for tmp in constant_temperatures:
        plt.plot(rewards_avgs[str(tmp)])
    plt.legend(['temp: ' + str(tmp) for tmp in constant_temperatures])
    plt.ylabel('Average Reward')
    plt.xlabel('Simulation Step')
    plt.grid()
    plt.title('Experiment 7: NaiivePreference - Varying Constant Temperatures')
    try:
        plt.savefig(os.path.join(d, f'rewards_{plot_root_name}.png'))
    except:
        print(f'Could not save rewards_{plot_root_name} plots')
    finally:
        plt.show()

    _ = plt.figure()
    for tmp in constant_temperatures:
        plt.plot(regret_averages[str(tmp)])
    plt.legend(['temp: ' + str(tmp) for tmp in constant_temperatures])
    plt.ylabel('Average Regret')
    plt.xlabel('Simulation Step')
    plt.grid()
    plt.title('Experiment 7: NaiivePreference - Varying Constant Temperatures')
    try:
        plt.savefig(os.path.join(d, f'regrets_{plot_root_name}.png'))
    except:
        print(f'Could not save regrets_{plot_root_name} plots')
    finally:
        plt.show()


def experiment_8(n_steps, n_trials):
    """
            Naiive preference policy over different alphas
    :return:
    """

    n_bandits = 10  # Each bandit is triggered by one action
    H0 = 0.0
    Rbar0 = 0.0
    alphas = [0.1, 0.4]
    reward_randomness_scale = 1.0
    constant_temperature = 0.1
    plot_root_text = 'experiment_8'

    test_bed_constructor = lambda: get_cont_reward_test_bed(
        reward_means=np.random.normal(0., 1., size=n_bandits).tolist(),
        reward_randomness_scale=[reward_randomness_scale] * n_bandits,
        stationary=True)

    policy_constructor = lambda _alpha: NaiivePreferencePolicy(
        n_actions=n_bandits,
        preference_initial_value=H0,
        reward_initial_value=Rbar0,
        learning_rate=_alpha,
        temperature=ConstantCoefficient(constant_temperature)
    )

    rewards, cummulative_rewards, cummulative_best_means, best_arms = simulate_paramterized_policy(
        test_bed_constructor=test_bed_constructor,
        policy_constructor=policy_constructor,
        varying_parameters=alphas,
        n_trials=n_trials,
        n_steps=n_steps,
        desc='Experiment 8, alphas')

    reward_avgs = dict()
    regret_averages = dict()
    for a in alphas:
        reward_avgs[str(a)] = [
            np.mean([rewards[str(a)][trial][step] for trial in range(n_trials)]) for step in range(n_steps)]

        regret_averages[str(a)] = [
            np.mean([cummulative_best_means[str(a)][trial][step] - cummulative_rewards[str(a)][trial][step]
                     for trial in range(n_trials)])
            for step in range(1, n_steps + 1)]

    d = os.path.join(os.getcwd(), 'plots')
    _ = mk_clear_dir(d, False)

    _ = plt.figure()
    for a in alphas:
        plt.plot(reward_avgs[str(a)])
    plt.legend(['alpha: ' + str(a) for a in alphas])
    plt.ylabel('Average Reward')
    plt.xlabel('Simulation Step')
    plt.title('Experiment 8: NaiivePreference - Varying Learning Rates (alpha)')
    try:
        plt.savefig(os.path.join(d, f'rewards_{plot_root_text}.png'))
    except:
        print(f'Could not save rewards_{plot_root_text} plots')
    finally:
        plt.show()

    _ = plt.figure()
    for a in alphas:
        plt.plot(regret_averages[str(a)])
    plt.legend(['alpha: ' + str(a) for a in alphas])
    plt.ylabel('Average Regret')
    plt.xlabel('Simulation Step')
    plt.title('Experiment 8: NaiivePreference - Varying Learning Rates (alpha)')
    try:
        plt.savefig(os.path.join(d, f'regrets_{plot_root_text}.png'))
    except:
        print(f'Could not save regrets_{plot_root_text} plots')
    finally:
        plt.show()


def experiment_9(n_steps, n_trials):
    """
        NaiivePreferencePolicy: Compare a decaying temperature vs. fixed
    :return:
    """
    n_bandits = 10  # Each bandit is triggered by one action
    H0 = 0.0
    Rbar0 = 0.0
    alpha = 0.1
    reward_randomness_scale = 1.0
    temperature = 1.0
    plot_root_text = 'experiment_9'

    test_bed_constructor = lambda: get_cont_reward_test_bed(
        reward_means=np.random.normal(0., 1., size=n_bandits).tolist(),
        reward_randomness_scale=[reward_randomness_scale] * n_bandits,
        stationary=True)

    # decaying temperature
    policy_constructor_dt = lambda _temp: NaiivePreferencePolicy(
        n_actions=n_bandits,
        preference_initial_value=H0,
        reward_initial_value=Rbar0,
        learning_rate=alpha,
        temperature=CosineDecayCoefficient(base_value=temperature, decay_steps=n_steps, final_value=1e-1)
    )

    rewards_dt, cummulative_rewards_dt, cummulative_best_means_dt, best_arms_dt = simulate_paramterized_policy(
        test_bed_constructor=test_bed_constructor,
        policy_constructor=policy_constructor_dt,
        varying_parameters=[temperature],
        n_trials=n_trials,
        n_steps=n_steps,
        desc='Experiment 9 - decaying temperature')

    rewards_avgs_dt = [np.mean([rewards_dt[str(temperature)][trial][step] for trial in range(n_trials)])
                       for step in range(n_steps)]

    regret_averages_dt = [
        np.mean([
            cummulative_best_means_dt[str(temperature)][trial][step] - cummulative_rewards_dt[str(temperature)][trial][
                step]
            for trial in range(n_trials)])
        for step in range(1, n_steps + 1)]

    # constant temperature
    policy_constructor_ct = lambda _temp: NaiivePreferencePolicy(
        n_actions=n_bandits,
        preference_initial_value=H0,
        reward_initial_value=Rbar0,
        learning_rate=alpha,
        temperature=ConstantCoefficient(base_value=temperature)
    )

    rewards_ct, cummulative_rewards_ct, cummulative_best_means_ct, best_arms_ct = simulate_paramterized_policy(
        test_bed_constructor=test_bed_constructor,
        policy_constructor=policy_constructor_ct,
        varying_parameters=[temperature],
        n_trials=n_trials,
        n_steps=n_steps,
        desc='Experiment 9 - constant temperature')

    rewards_avgs_ct = [np.mean([rewards_ct[str(temperature)][trial][step] for trial in range(n_trials)])
                       for step in range(n_steps)]

    regret_averages_ct = [
        np.mean([
            cummulative_best_means_ct[str(temperature)][trial][step] - cummulative_rewards_ct[str(temperature)][trial][
                step]
            for trial in range(n_trials)])
        for step in range(1, n_steps + 1)]

    d = os.path.join(os.getcwd(), 'plots')
    _ = mk_clear_dir(d, False)

    _ = plt.figure()
    plt.plot(rewards_avgs_dt)
    plt.plot(rewards_avgs_ct)
    plt.legend(['Temp: 2 -> 0.1', 'Temp: 2'])
    plt.ylabel('Average Reward')
    plt.xlabel('Simulation Step')
    plt.title('Experiment 9: NaiivePreference - Cosine-decaying vs Constant Temp')
    try:
        plt.savefig(os.path.join(d, f'rewards_{plot_root_text}.png'))
    except:
        print(f'Could not save rewards_{plot_root_text} plots')
    finally:
        plt.show()

    _ = plt.figure()
    plt.plot(regret_averages_dt)
    plt.plot(regret_averages_ct)
    plt.legend(['Temp: 2 -> 0.1', 'Temp: 2'])
    plt.ylabel('Average Regret')
    plt.xlabel('Simulation Step')
    plt.title('Experiment 9: NaiivePreference - Cosine-decaying vs Constant Temp')
    try:
        plt.savefig(os.path.join(d, f'regrets_{plot_root_text}.png'))
    except:
        print(f'Could not save regrets_{plot_root_text} plots')
    finally:
        plt.show()


def experiment_10(n_steps, n_trials):
    """
            Naiive preference policy over different learning rates with/out baseline
            Repeating the experiment in fig 2.5 in Sutton book
    :return:
    """

    n_bandits = 10  # Each bandit is triggered by one action
    H0 = 0.0
    Rbar0 = 0.0
    alphas = [0.1, 0.4]
    reward_randomness_scale = 1.0
    temperature = 1.0
    plot_root_text = 'experiment_10'

    test_bed_constructor = lambda: get_cont_reward_test_bed(
        reward_means=np.random.normal(4.0, 1.0, size=n_bandits).tolist(),
        reward_randomness_scale=[reward_randomness_scale] * n_bandits,
        stationary=True)

    policy_constructor_with_baseline = lambda _alpha: NaiivePreferencePolicy(
        n_actions=n_bandits,
        preference_initial_value=H0,
        reward_initial_value=Rbar0,
        learning_rate=_alpha,
        temperature=ConstantCoefficient(temperature),
        use_baseline=True
    )

    rewards_with_baseline, \
    cummulative_rewards_with_baseline, \
    cummulative_best_means_with_baseline, \
    best_arms_with_baseline = simulate_paramterized_policy(
        test_bed_constructor=test_bed_constructor,
        policy_constructor=policy_constructor_with_baseline,
        varying_parameters=alphas,
        n_trials=n_trials,
        n_steps=n_steps,
        desc='Experiment 10 - with baseline')

    reward_avgs_with_baseline = dict()
    regret_averages_with_baseline = dict()
    for alpha in alphas:
        reward_avgs_with_baseline[str(alpha)] = [
            np.mean([rewards_with_baseline[str(alpha)][trial][step] for trial in range(n_trials)]) for step in
            range(n_steps)]

        regret_averages_with_baseline[str(alpha)] = [
            np.mean([
                cummulative_best_means_with_baseline[str(alpha)][trial][step] -
                cummulative_rewards_with_baseline[str(alpha)][trial][
                    step]
                for trial in range(n_trials)])
            for step in range(1, n_steps + 1)]

    policy_constructor_no_baseline = lambda _alpha: NaiivePreferencePolicy(
        n_actions=n_bandits,
        preference_initial_value=H0,
        reward_initial_value=Rbar0,
        learning_rate=_alpha,
        temperature=ConstantCoefficient(temperature),
        use_baseline=False
    )

    rewards_no_baseline, \
    cummulative_rewards_no_baseline, \
    cummulative_best_means_no_baseline, \
    best_arms_no_baseline = simulate_paramterized_policy(
        test_bed_constructor=test_bed_constructor,
        policy_constructor=policy_constructor_no_baseline,
        varying_parameters=alphas,
        n_trials=n_trials,
        n_steps=n_steps,
        desc='Experiment 10 - no baseline')

    reward_avgs_no_baseline = dict()
    regret_averages_no_baseline = dict()
    for alpha in alphas:
        reward_avgs_no_baseline[str(alpha)] = [
            np.mean([rewards_no_baseline[str(alpha)][trial][step] for trial in range(n_trials)]) for step in
            range(n_steps)]
        regret_averages_no_baseline[str(alpha)] = [
            np.mean([
                cummulative_best_means_no_baseline[str(alpha)][trial][step] -
                cummulative_rewards_no_baseline[str(alpha)][trial][
                    step]
                for trial in range(n_trials)])
            for step in range(1, n_steps + 1)]

    d = os.path.join(os.getcwd(), 'plots')
    _ = mk_clear_dir(d, False)

    _ = plt.figure()
    legend = []
    for alpha in alphas:
        plt.plot(reward_avgs_with_baseline[str(alpha)])
        legend.append('Baseline - alpha: ' + str(alpha))
        plt.plot(reward_avgs_no_baseline[str(alpha)])
        legend.append('No baseline - alpha: ' + str(alpha))
    plt.legend(legend)
    plt.ylabel('Average Reward')
    plt.xlabel('Simulation Step')
    plt.title('Experiment 10: NaiivePreference - Vary LR and Baseline')
    try:
        plt.savefig(os.path.join(d, f'rewards_{plot_root_text}.png'))
    except:
        print(f'Could not save rewards_{plot_root_text} plots')
    finally:
        plt.show()

    _ = plt.figure()
    legend = []
    for alpha in alphas:
        plt.plot(regret_averages_with_baseline[str(alpha)])
        legend.append('Baseline - alpha: ' + str(alpha))
        plt.plot(regret_averages_no_baseline[str(alpha)])
        legend.append('No baseline - alpha: ' + str(alpha))
    plt.legend(legend)
    plt.ylabel('Average Regret')
    plt.xlabel('Simulation Step')
    plt.title('Experiment 10: NaiivePreference - Vary LR and Baseline')
    try:
        plt.savefig(os.path.join(d, f'regrets_{plot_root_text}.png'))
    except:
        print(f'Could not save regrets_{plot_root_text} plots')
    finally:
        plt.show()


def experiment_11(n_steps, n_trials):
    """
        Run SoftmaxExploration policy over varying fixed temperatures
    :return:
    """

    n_bandits = 10  # Each bandit is triggered by one action
    Q0 = 0.0
    reward_randomness_scale = 1.0
    temperatures = [0.1, 0.2, 0.5, 1.0, 2.0]
    plot_root_text = 'experiment_11'

    test_bed_constructor = lambda: get_cont_reward_test_bed(
        reward_means=np.random.normal(0, 1., size=n_bandits).tolist(),
        reward_randomness_scale=[reward_randomness_scale] * n_bandits,
        stationary=True)

    q_constructor = lambda: QMonteCarlo(n_actions=n_bandits, initial_action_value=Q0)
    policy_constructor = lambda _temp: SoftmaxExplorationPolicy(q=q_constructor(), temperature=_temp)

    rewards, cummulative_rewards, cummulative_best_means, best_arms = simulate_paramterized_policy(
        test_bed_constructor=test_bed_constructor,
        policy_constructor=policy_constructor,
        varying_parameters=temperatures,
        n_trials=n_trials,
        n_steps=n_steps,
        desc='Experiment 11 - SoftmaxExploration, temperatures')

    reward_avgs = dict()
    regret_avgs = dict()
    for temp in temperatures:
        reward_avgs[str(temp)] = [
            np.mean([rewards[str(temp)][trial][step] for trial in range(n_trials)]) for step in range(n_steps)]

        regret_avgs[str(temp)] = [
            np.mean([
                cummulative_best_means[str(temp)][trial][step] -
                cummulative_rewards[str(temp)][trial][
                    step]
                for trial in range(n_trials)])
            for step in range(1, n_steps + 1)]

    d = os.path.join(os.getcwd(), 'plots')
    _ = mk_clear_dir(d, False)

    _ = plt.figure()
    for temp in temperatures:
        plt.plot(reward_avgs[str(temp)])
    plt.legend(['temp: ' + str(t) for t in temperatures])
    plt.ylabel('Average Reward')
    plt.xlabel('Simulation Step')
    plt.title('Experiment 11: SoftmaxExploration, Non-Stationary Env.- \nVarying Temperatures')
    try:
        plt.savefig(os.path.join(d, f'rewards_{plot_root_text}.png'))
    except:
        print(f'Could not save rewards_{plot_root_text} plots')
    finally:
        plt.show()

    _ = plt.figure()
    for temp in temperatures:
        plt.plot(regret_avgs[str(temp)])
    plt.legend(['temp: ' + str(t) for t in temperatures])
    plt.ylabel('Average Regret')
    plt.xlabel('Simulation Step')
    plt.title('Experiment 11: SoftmaxExploration, Non-Stationary Env.- \nVarying Temperatures')
    try:
        plt.savefig(os.path.join(d, f'regrets_{plot_root_text}.png'))
    except:
        print(f'Could not save regrets_{plot_root_text} plots')
    finally:
        plt.show()


######### Binary valued reward (Bernoulli TestBed) #############

def experiment_12(n_steps, n_trials):
    """
        GreedyBernoulli vs. Bernoulli Thompson Sampling on stationary Bernoulli testbed
    """
    n_bandits = 10  # Each bandit is triggered by one action
    min_success = 0.1
    max_successs = 0.9
    plot_root_name = 'experiment_12'
    sentinel = 'none'

    true_success_rates = lambda: [float(np.random.uniform(min_success, max_successs, size=1)) for _ in range(n_bandits)]
    test_bed_constructor = lambda: get_binary_reward_test_bed(
        success_rates=true_success_rates(),
        stationary=True)

    greedy_policy_constructor = lambda: BernoulliGreedy(n_actions=n_bandits, initial_alpha=1., initial_beta=1.)

    rewards, cummulative_rewards, cummulative_best_means, best_arms = simulate_bernoulli_testbed(
        test_bed_constructor=test_bed_constructor,
        policy_constructor=greedy_policy_constructor)

    greedy_reward_averages = dict()
    greedy_regret_averages = dict()

    greedy_reward_averages[sentinel] = [
        np.mean([rewards[sentinel][trial][step] for trial in range(n_trials)]) for step in range(n_steps)]

    greedy_regret_averages[sentinel] = [
        np.mean([(cummulative_best_means[sentinel][trial][step] - cummulative_rewards[sentinel][trial][step]) / step
                 for trial in range(n_trials)])
        for step in range(1, n_steps + 1)]


    # Thompson sampling
    ts_policy_constructor = lambda: BernoulliThompsonSampling(n_actions=n_bandits, initial_alpha=1., initial_beta=1.)

    rewards, cummulative_rewards, cummulative_best_means, best_arms = simulate_bernoulli_testbed(
        test_bed_constructor=test_bed_constructor,
        policy_constructor=ts_policy_constructor)

    ts_reward_averages = dict()
    ts_regret_averages = dict()

    ts_reward_averages[sentinel] = [
        np.mean([rewards[sentinel][trial][step] for trial in range(n_trials)]) for step in range(n_steps)]

    ts_regret_averages[sentinel] = [
        np.mean([(cummulative_best_means[sentinel][trial][step] - cummulative_rewards[sentinel][trial][step]) / step
                 for trial in range(n_trials)])
        for step in range(1, n_steps + 1)]

    d = os.path.join(os.getcwd(), 'plots')
    _ = mk_clear_dir(d, False)

    _ = plt.figure()
    plt.plot(greedy_reward_averages[sentinel])
    plt.plot(ts_reward_averages[sentinel])
    plt.legend(['Greedy', 'Thompson-Sampling'])
    plt.ylabel('Average Reward')
    plt.xlabel('Simulation Step')
    plt.title('Experiment 12: Bernoulli Greedy vs Thompson Sampling\n Avg. Rewards')
    plt.grid()
    try:
        plt.savefig(os.path.join(d, f'rewards_{plot_root_name}.png'))
    except:
        print(f'Could not save rewards_{plot_root_name} plot')
    finally:
        plt.show()

    _ = plt.figure()
    plt.plot(greedy_regret_averages[sentinel])
    plt.plot(ts_regret_averages[sentinel])
    plt.legend(['Greedy', 'Thompson-Sampling'])
    plt.ylabel('Average Regret')
    plt.xlabel('Simulation Step')
    plt.title('Experiment 12: Bernoulli Greedy vs Thompson Sampling\nAvg. Regrets')
    plt.grid()
    try:
        plt.savefig(os.path.join(d, f'regrets_{plot_root_name}.png'))
    except:
        print(f'Could not save regrets_{plot_root_name} plot')
    finally:
        plt.show()


def experiment_13(n_steps, n_trials):
    """
       Bernoulli Thompson Sampling over a stationary and non-stationary Bernoulli testbed
    """
    n_bandits = 10  # Each bandit is triggered by one action
    min_success = 0.1
    max_successs = 0.9
    plot_root_name = 'experiment_13'
    sentinel = 'none'
    rand_scale = 0.02

    true_success_rates = lambda: [float(np.random.uniform(min_success, max_successs, size=1)) for _ in range(n_bandits)]
    reward_randomness_scales = [rand_scale for _ in range(n_bandits)]
    stationary_test_bed_constructor = lambda: get_binary_reward_test_bed(
        success_rates=true_success_rates(),
        stationary=True)

    non_stationary_test_bed_constructor = lambda: get_binary_reward_test_bed(
        success_rates=true_success_rates(),
        reward_randomness_scales=reward_randomness_scales,
        stationary=False)

    policy_constructor = lambda: BernoulliThompsonSampling(n_actions=n_bandits, initial_alpha=1., initial_beta=1.)

    rewards, cummulative_rewards, cummulative_best_means, best_arms = simulate_bernoulli_testbed(
        test_bed_constructor=stationary_test_bed_constructor,
        policy_constructor=policy_constructor)

    stationary_reward_averages = dict()
    stationary_regret_averages = dict()

    stationary_reward_averages[sentinel] = [
        np.mean([rewards[sentinel][trial][step] for trial in range(n_trials)]) for step in range(n_steps)]

    stationary_regret_averages[sentinel] = [
        np.mean([(cummulative_best_means[sentinel][trial][step] - cummulative_rewards[sentinel][trial][step]) / step
                 for trial in range(n_trials)])
        for step in range(1, n_steps + 1)]


    rewards, cummulative_rewards, cummulative_best_means, best_arms = simulate_bernoulli_testbed(
        test_bed_constructor=non_stationary_test_bed_constructor,
        policy_constructor=policy_constructor)

    non_stationary_reward_averages = dict()
    non_stationary_regret_averages = dict()

    non_stationary_reward_averages[sentinel] = [
        np.mean([rewards[sentinel][trial][step] for trial in range(n_trials)]) for step in range(n_steps)]

    non_stationary_regret_averages[sentinel] = [
        np.mean([(cummulative_best_means[sentinel][trial][step] - cummulative_rewards[sentinel][trial][step]) / step
                 for trial in range(n_trials)])
        for step in range(1, n_steps + 1)]

    d = os.path.join(os.getcwd(), 'plots')
    _ = mk_clear_dir(d, False)

    _ = plt.figure()
    plt.plot(stationary_reward_averages[sentinel])
    plt.plot(non_stationary_reward_averages[sentinel])
    plt.legend(['Stationary', 'Non-Stationary'])
    plt.ylabel('Average Reward')
    plt.xlabel('Simulation Step')
    plt.title('Experiment 13: Thompson Sampling: Stationary vs Non-Stationary Environment\n Avg. Rewards')
    plt.grid()
    try:
        plt.savefig(os.path.join(d, f'rewards_{plot_root_name}.png'))
    except:
        print(f'Could not save rewards_{plot_root_name} plot')
    finally:
        plt.show()

    _ = plt.figure()
    plt.plot(stationary_regret_averages[sentinel])
    plt.plot(non_stationary_regret_averages[sentinel])
    plt.legend(['Stationary', 'Non-Stationary'])
    plt.ylabel('Average Regret')
    plt.xlabel('Simulation Step')
    plt.title('Experiment 13: Thompson Sampling: Stationary vs Non-Stationary Environment\nAvg. Regrets')
    plt.grid()
    try:
        plt.savefig(os.path.join(d, f'regrets_{plot_root_name}.png'))
    except:
        print(f'Could not save regrets_{plot_root_name} plot')
    finally:
        plt.show()


n_steps, n_trials = 1000, 500

experiment_13(n_steps, n_trials)

funs = locals()
funs2exec = list()

for e in range(1, 12):
    fun_name = "experiment_" + str(e)
    if fun_name in funs:
        funs2exec.append(funs[fun_name])
if len(funs2exec) > 0:
    results = Parallel(n_jobs=5)(delayed(lambda _f: _f(n_steps, n_trials))(f) for f in funs2exec)
else:
    print('No functions to process in parallel')

exit(0)
