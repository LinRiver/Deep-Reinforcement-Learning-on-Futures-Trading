# Deep-reinforcement-Learning-on-Futures-Trading
## Introduction
Based on Reinforcement Learning: Q-Learning with Clockwork RNN we develop the Futures-Trading-Robot.
There are four main features below,
1. Clockwork RNN is modified to two hidden layers.
2. We adopt Double Dueling-DQN instead of DQN to improve the robustness of trading performance.
3. For exploration and exploitation on model stability as training, boltzmann_policy is better than greedy_policy.
4. Considering time series, data dependency, we replace experience replay with data incremental method.

## Reference
1. A clockwork RNN, https://arxiv.org/abs/1402.3511
2. Dueling Network Architectures for Deep Reinforcement Learning, http://proceedings.mlr.press/v48/wangf16.pdf
3. Exploration in DeepReinforcement Learning, https://www.ias.informatik.tu-darmstadt.de/uploads/Theses/Abschlussarbeiten/markus_semmler_bsc.pdf
