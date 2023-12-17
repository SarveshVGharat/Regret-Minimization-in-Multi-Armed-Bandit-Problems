# Regret Minimization in Multi-Armed Bandit Problems

In the realm of decision-making under uncertainty, multi-armed bandit problems serve as a quintessential challenge. Our project was dedicated to devising effective strategies for minimizing regret – the difference between the reward accumulated by an algorithm and the maximum achievable reward. Here are the key highlights of the work

1. Algorithm Diversity: The project confirms the theoretic logarithmic upper bounds on regret for different algorithms including UCB (Upper Confidence Bound), KLUCB (Kullback-Leibler Upper Confidence Bound), and Thompson Sampling. These algorithms formed the backbone of our approach, enabling us to explore various avenues for regret minimization.
2. Batch Thompson Sampling: An innovative aspect of the project involved the implementation of Thompson Sampling in a batch setting. This adaptation was aimed at minimizing regret when samples are taken in batches.
3. Addressing Special Cases: The project addressed a unique scenario in bandit problems where the number of arms matches the horizon. In such cases, we provided a straightforward yet effective approach to minimize regret, ensuring that even under these specific constraints, our strategies remained optimized.
