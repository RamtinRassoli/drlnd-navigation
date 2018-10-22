# Report

## Learning Algorithm
The reinforcement learning algorithm implemented in the project is [Double DQ-learning](https://arxiv.org/pdf/1509.06461.pdf).

### Action Value Network Architecture

| Layer Name      | Layer Type,| Input Dimension          | Ouput  |
| -------------|:-------:|:-------------:| -----:|
| input      | Linear with Relu|(37,) | (128,) |
| dropout      | Dropout |(128,) | (128,) |
| hidden      | Linear with Relu|(128,) | (64,) |
| output | Linear |(64,) | (4,) |



### Hyperparameters
```yaml
exp_replay:
    buffer: 5000 # replay buffer size
    batch: 64 # minibatch size
    update_every: 4 # how often to update the network
dq:
    gamma: 0.99 # discount factor
    tau: 1e-3 # for soft update of target parameters
    lr: 5e-4 # learning rate
train:
    n_episodes: 2500
    max_t: 1000
    eps_start: 1.0
    eps_end: 1e-2
    eps_decay: 0.999
```

## Results
![](images/2000.png)
```
Episode 100	Average Score: 0.22
Episode 200	Average Score: 1.08
Episode 300	Average Score: 3.12
Episode 400	Average Score: 4.50
Episode 500	Average Score: 5.97
Episode 600	Average Score: 7.97
Episode 700	Average Score: 8.41
Episode 800	Average Score: 9.51
Episode 900	Average Score: 10.02
Episode 1000	Average Score: 10.34
Episode 1100	Average Score: 10.75
Episode 1200	Average Score: 12.60
Episode 1300	Average Score: 12.16
Episode 1400	Average Score: 11.74
Episode 1500	Average Score: 12.06
Episode 1600	Average Score: 12.72
Episode 1700	Average Score: 12.72
Environment solved in 1623 episodes!	Average Score: 13.06
```
## Ideas for Future Work

