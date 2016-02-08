# Introduction

Python implementation of policy gradient reinforcement learning methods, built on TensorFlow. 

Flexible reinforcement learning framework that does not require dynamics of the environment. 

Given a sequence of state-action pairs and rewards, adjusts policy based on gradient steps with respect to its parameters. 

Interface designed to minimize the amount of developer interaction needed. 

# Installation
Quick installation using the pip package manager.
> pip install -i https://testpypi.python.org/pypi policy_gradient

# Interface

Quick, clean interface for training a neural network using policy gradient methods.

Designed to minimize the amount of interaction with the interface. 

Just initialize the learner, pass in the dynamics/rewards functions and the initial state,
and you're good to go!

```python
# Initialize learner
learner = PolicyGradient(net_dims, 'tanh')
# Train policy
learner.train_agent(dynamics_func, reward_func, initial_state)
# Retrieve predicted best actions based on learned policy
learner.get_action(new_state)
```

# Documentation
[Full Documentation](http://wesleyhsieh.github.io/policy_gradient/)


