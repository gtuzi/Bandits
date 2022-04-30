
[Sutton & Barto RL Book]: http://incompleteideas.net/book/RLbook2020.pdf
[Algorithms]: https://arxiv.org/pdf/1402.6028.pdf
[Thompson Sampling Tutorial]: https://web.stanford.edu/~bvr/pubs/TS_Tutorial.pdf


# Non-Assocative Bandits
Implementation of different bandits algorithms in policies operating in a k-armed test bed. 

<img width="810" alt="Screen Shot 2022-04-25 at 1 51 36 AM" src="https://user-images.githubusercontent.com/3085599/165035548-95a25c07-6f4d-40ec-bad4-6e2bd51d6a78.png">

Two variants of the test-bed have been implemented
* Stationary: each bandit has fixed mean / std dev.
* Non-stationary: each bandit's mean follows a (normal) random walk 



Experiment run script is located in `/scripts`


## Non-assocative Algorithms

### Epsilon Greedy
An action value policy using an estimate of reward Q(a) to determine exploration / exploitation

<img width="539" alt="Screen Shot 2022-04-25 at 1 34 51 AM" src="https://user-images.githubusercontent.com/3085599/165033244-e70b67dd-9da7-415c-8ee2-7ce191646b71.png">

[Sutton & Barto RL Book]

* Experiment 1: For a fixed action-value (E[R | a]) for each bandit, compare the performance of epsilon greedy policies accross different values of exploration. Q uses sample averaging (unbiased estimator)
* Experiment 2: Compare the performances across different levels of reward noise
* Experiment 3: Compare the performances between the sample average (MC) Q and constant stepsize Q on a **stationary** testbed
* Experiment 4: Compare the performances between the sample average (MC) Q and constant stepsize Q on a **non-stationary** testbed
* Experiment 5: Compare pessimistic vs optimistic initial value function (Q)

<img src="/scripts/plots/rewards_experiment_1.png" width="400" height="300" />|<img src="/scripts/plots/rewards_experiment_2.png" width="400" height="300" />
<img src="/scripts/plots/rewards_experiment_3.png" width="400" height="300" />|<img src="/scripts/plots/rewards_experiment_4.png" width="400" height="300" />|
<img src="/scripts/plots/rewards_experiment_5.png" width="400" height="300" />|

---

### Action-Value Policy: UCB1
An action value policy using an estimate of reward Q(a) to determine exploration / exploitation

<img width="388" alt="Screen Shot 2022-04-25 at 1 35 53 AM" src="https://user-images.githubusercontent.com/3085599/165033407-496ac68f-8146-4384-9447-f654610fa64f.png">

[Sutton & Barto RL Book], [Algorithms]
* Experiment 6: Compare UCB1 policy with eps-greedy on a stationary test bed


<img src="/scripts/plots/rewards_experiment_6.png" width="400" height="300" />|

---

### Naiive preference
Naiive preference is a gradient-derived policy.
        
        Action is sampled as a ~ softmax(H)
        
<img width="402" alt="Screen Shot 2022-04-25 at 1 39 29 AM" src="https://user-images.githubusercontent.com/3085599/165033955-b9e8bc5f-2c5b-4bb5-8a6c-6e25016c9871.png">

        From eq (2.12) in Sutton book, 2nd edition (2018):
        
        H(A, t+1) = H(A, t) + alpha * Advantage * (1 - pi(A))
        H(o, t+1) = H(o, t) - alpha * Advantage * p(o)
        where:
        H: preference model
        R_bar: baseline, a moving average of reward received
        Advantage: R(t) - R_bar

<img width="776" alt="Screen Shot 2022-04-25 at 1 40 04 AM" src="https://user-images.githubusercontent.com/3085599/165034010-cac94ba4-994e-4e4a-951d-5ac391250bbf.png">

  
[Sutton & Barto RL Book]

* Experiment 7: Naiive preference policy over different constant temperatures
* Experiment 8: Naiive preference policy over different alphas
* Experiment 9: NaiivePreferencePolicy: Compare a decaying temperature vs. fixed
* Experiment 10: Naiive preference policy over different learning rates with/out baseline. Repeating the experiment in fig 2.5 in Sutton book

<img src="/scripts/plots/rewards_experiment_7.png" width="400" height="300" />|<img src="/scripts/plots/rewards_experiment_8.png" width="400" height="300" />
<img src="/scripts/plots/rewards_experiment_9.png" width="400" height="300" />|<img src="/scripts/plots/rewards_experiment_10.png" width="400" height="300" />

---

### Softmax / Boltzman Exploration
Softmax methods are based on Luce’s axiom of choice (1959) and pick each arm with a probability that is proportional to its average reward. Arms with greater empirical means are therefore picked with higher probability. Alternative name, Boltzman exploration

<img width="393" alt="Screen Shot 2022-04-25 at 1 45 27 AM" src="https://user-images.githubusercontent.com/3085599/165034721-5b057b73-6193-44f2-84b8-e1dabb3be866.png">


[Algorithms]

* Experiment 11: Run SoftmaxExploration policy over varying fixed temperatures

<img src="/scripts/plots/rewards_experiment_11.png" width="400" height="300" />|

---

## Thompson Sampling
_Dithering_ is a common approach to exploration that operates through randomly perturbing actions that would be selected by a greedy algorithm, such as eps-greedy. Eps-greedy exploration would allocate an equal number of experimental trials to each action. Though only half of the exploratory actions are wasted in this example, the issue is exacerbated as the number of possible actions increases. Thompson Sampling provides an alternative to dithering that more intelligently allocates exploration effort. 

Let the agent begin with an independent prior belief over each _θk_. Take these priors to be beta-distributed with parameters _α = (α1, . . . , αK)_ and _β ∈ (β1, . . . , βK)_. In particular, for each action _k_, the prior probability density function of _θk_ is

<img width="666" alt="Screen Shot 2022-04-30 at 4 31 21 PM" src="https://user-images.githubusercontent.com/3085599/166123362-9386e771-d62a-453f-ba2b-3710c2a85b6b.png">

As observations are gathered, the distribution is updated according to Bayes’ rule. It is particularly convenient to work with beta distributions because of their conjugacy properties. In particular, each action’s posterior distribution is also beta with parameters that can be updated according to a simple rule:

<img width="816" alt="Screen Shot 2022-04-30 at 4 34 35 PM" src="https://user-images.githubusercontent.com/3085599/166123428-9d29ebbf-42a7-47f1-b23d-733667b97685.png">

Note that for the special case of _αk_ = _βk_ = 1, the prior _p(θk)_ is uniform over [0, 1]. Note that only the parameters of a __selected action__ are updated. A beta distribution with parameters _(αk, βk)_ has mean _αk/(αk + βk)_, and the distribution becomes more concentrated as _αk + βk_ grows.

___Algorithm 1___ presents a __greedy__ algorithm for the beta-Bernoulli bandit. In each time period _t_, the algorithm generates an estimate _ˆθk = αk/(αk + βk)_, equal to its current expectation of the success probability _θk_. The action _xt_ with the largest estimate _ˆθk_ is then applied, after which a reward _rt_ is observed and the distribution parameters _αxt_ and _βxt_ are updated.

___Algorithm 2___  - Thompson Sampling - is specialized to the case of a beta-Bernoulli bandit, and it is similart to _Algorithm 1_. The only difference is that the success probability estimate _ˆθk_ is _randomly_ sampled from the posterior distribution, which is a beta distribution with parameters _αk_ and _βk_, rather than taken to be the expectation _αk/(αk+βk)_.


<img width="1134" alt="Screen Shot 2022-04-30 at 4 43 52 PM" src="https://user-images.githubusercontent.com/3085599/166123621-c67671d7-2178-4c98-8ecb-ac3179621c76.png">

Refer to [Thompson Sampling Tutorial] for details.
_Note:_ for the TS experiments, a Bernoulli test bed was used, with different success rates for each arm.

* Experiment 11: GreedyBernoulli vs. Bernoulli Thompson Sampling on stationary Bernoulli testbed
* Experiment 12: Bernoulli Thompson Sampling over a stationary and non-stationary Bernoulli testbed

<img src="/scripts/plots/rewards_experiment_12.png" width="400" height="300" />|<img src="/scripts/plots/rewards_experiment_13.png" width="400" height="300" />
<img src="/scripts/plots/regrets_experiment_12.png" width="400" height="300" />|<img src="/scripts/plots/regrets_experiment_13.png" width="400" height="300" />

---
