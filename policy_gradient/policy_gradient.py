import numpy as np
import numpy.linalg as la
import collections
import IPython
import tensorflow as tf
from utils import *
import time
from collections import defaultdict

class PolicyGradient(Utils):

	""" 
	Calculates policy gradient
	for given input state/actions.

	Users should primarily be calling main
	PolicyGradient class methods.
	"""

	def __init__(self, net_dims, filepath=None, q_net_dims=None, output_function=None, seed=0, seed_state=None):
		"""
		Initializes PolicyGradient class.

		Parameters:
		net_dims: array-like
			1D list corresponding to dimensions
			of each layer in the net.
		output_function: string
			Non-linearity function applied to output of
			neural network. 
			Options are: 'tanh', 'sigmoid', 'relu', 'softmax'.
		"""
		self.q_dict = defaultdict(lambda: defaultdict(float))
		self.prev_weight_grad = self.prev_bias_grad = self.prev_weight_update_vals = \
		self.prev_bias_update_vals = self.prev_weight_inverse_hess = self.prev_bias_inverse_hess = \
		self.total_weight_grad = self.total_bias_grad = None

		self.init_action_neural_net(net_dims, output_function, filepath)
		if seed_state is not None:
			np.random.set_state(seed_state)
		tf.set_random_seed(seed)


	def train_agent(self, dynamics_func, reward_func, update_method, initial_state, num_iters, batch_size, traj_len, step_size=0.1, momentum=0.5, normalize=True):
		"""
		Trains agent using input dynamics and rewards functions.

		Parameters:
		dynamics_func: function
			User-provided function that takes in 
			a state and action, and returns the next state.
		reward_func: function
			User-provided function that takes in 
			a state and action, and returns the associated reward.
		initial_state: array-like
			Initial state that each trajectory starts at.
			Must be 1-dimensional NumPy array.
		num_iters: int
			Number of iterations to run gradient updates.
		batch_size: int
			Number of trajectories to run in a single iteration.
		traj_len: int
			Number of state-action pairs in a trajectory.

		Output:
		mean_rewards: array-like
			Mean ending rewards of all iterations.
		"""
		mean_rewards = []
		ending_states = []
		for i in range(num_iters):
			traj_states = []
			traj_actions = []
			rewards = []

			for j in range(batch_size):
				states = []
				actions = []
				curr_rewards = []
				curr_state = initial_state
				# Rolls out single trajectory
				for k in range(traj_len):
					# Get action from learner
					curr_action = self.get_action(curr_state)

					# Update values
					states.append(curr_state)
					curr_rewards.append(reward_func(curr_state, curr_action))
					actions.append(curr_action)

					# Update state
					curr_state = dynamics_func(curr_state, curr_action)

				# Append trajectory/rewards
				traj_states.append(states)
				traj_actions.append(actions)
				rewards.append(curr_rewards)

			# Apply policy gradient iteration
			self.gradient_update(np.array(traj_states), np.array(traj_actions), np.array(rewards), \
					update_method, step_size, momentum, normalize)
			mean_rewards.append(np.mean([np.sum(reward_list) for reward_list in rewards]))
			ending_states.append([traj[-1] for traj in traj_states])
		return np.array(mean_rewards), ending_states

	def gradient_update(self, traj_states, traj_actions, rewards, update_method='sgd', step_size=1.0, momentum=0.5, normalize=True):
		"""
		Estimates and applies gradient update according to a policy.

		States, actions, rewards must be lists of lists; first dimension indexes
		the ith trajectory, second dimension indexes the jth state-action-reward of that
		trajectory.
		
		Parameters:
		traj_states: array-like
			List of list of states.
		traj_actions: array-like
			List of list of actions.
		rewards: array-like
			List of list of rewards.
		step_size: float
			Step size.
		momentum: float
			Momentum value.
		normalize: boolean
			Determines whether to normalize gradient update. 
			Recommended if running into NaN/infinite value errors.
		"""
		assert update_method in ['sgd', 'momentum', 'lbfgs', 'adagrad', 'rmsprop', 'adam']
		# Calculate updates and create update pairs
		curr_weight_grad = 0
		curr_bias_grad = 0
		curr_weight_update_vals = []
		curr_bias_update_vals = []
		curr_weight_inverse_hess = []
		curr_bias_inverse_hess = []

		iters = traj_states.shape[0]
		q_vals = self.estimate_q(traj_states, traj_actions, rewards)
		assert traj_states.shape[0] == traj_actions.shape[0] == rewards.shape[0]
		assert q_vals.shape[0] == iters

		# Update for each example
		for i in range(iters):
			# Estimate q-values and extract gradients
			curr_traj_states = traj_states[i]
			curr_traj_actions = traj_actions[i]
			curr_q_val_list = q_vals[i]
			curr_traj_states = curr_traj_states.reshape(curr_traj_states.shape[0], curr_traj_states.shape[1] * curr_traj_states.shape[2])
			curr_traj_actions = curr_traj_actions.reshape(curr_traj_actions.shape[0], curr_traj_actions.shape[1] * curr_traj_actions.shape[2])
			curr_q_val_list = curr_q_val_list.reshape(curr_q_val_list.shape[0], 1)

			curr_weight_grad_vals = self.sess.run(self.weight_grads, \
					feed_dict={self.input_state: curr_traj_states, self.observed_action: curr_traj_actions, self.q_val: curr_q_val_list})
			curr_bias_grad_vals = self.sess.run(self.bias_grads, \
					feed_dict={self.input_state: curr_traj_states, self.observed_action: curr_traj_actions, self.q_val: curr_q_val_list})

			curr_weight_grad += np.array(curr_weight_grad_vals) / np.float(iters)
			curr_bias_grad += np.array(curr_bias_grad_vals) / np.float(iters)

		# Update weights
		for j in range(len(self.weights)):
			if update_method == 'sgd':
				update_val = step_size * curr_weight_grad[j]

			elif update_method == 'momentum':
				if self.prev_weight_grad is None:
					update_val = step_size * curr_weight_grad[j]
				else:
					update_val = momentum * self.prev_weight_grad[j] +  step_size * curr_weight_grad[j]

			elif update_method == 'lbfgs':
				if self.prev_weight_inverse_hess is None:
					curr_inverse_hess = np.eye(curr_weight_grad[j].shape[0])
					update_val = curr_weight_grad[j]
				else: 
					update_val, curr_inverse_hess = \
						self.bfgs_update(self.prev_inverse_hess[j], self.prev_update_val[j], self.prev_weight_grad[j], update_val)
				update_val = update_val * step_size 
				curr_weight_inverse_hess.append(curr_inverse_hess)

			elif update_method == 'adagrad':
				if self.total_weight_grad is None:
					self.total_weight_grad = curr_weight_grad
				else:
					self.total_weight_grad[j] += np.square(curr_weight_grad[j])
				update_val = step_size * curr_weight_grad[j] / (np.sqrt(np.abs(self.total_weight_grad[j])) + 1e-8)

			elif update_method == 'rmsprop':
				decay = 0.99
				if self.total_weight_grad is None:
					self.total_weight_grad = curr_weight_grad
				else:
					self.total_weight_grad[j] = decay * self.total_weight_grad[j] + (1 - decay) * np.square(curr_weight_grad[j])
				update_val = step_size * curr_weight_grad[j] / (np.sqrt(np.abs(self.total_weight_grad[j])) + 1e-8)

			elif update_method == 'adam':
				beta1, beta2 = 0.9, 0.999
				if self.total_weight_grad is None:
					self.total_weight_grad = curr_weight_grad
					self.total_sq_weight_grad = np.square(curr_weight_grad)
				else:
					self.total_weight_grad[j] = beta1 * self.total_weight_grad[j] + (1 - beta1) * curr_weight_grad[j]
					self.total_sq_weight_grad[j] = beta2 * self.total_sq_weight_grad[j] + (1 - beta2) * np.sqrt(np.abs(self.total_weight_grad[j]))
				update_val = np.divide(step_size * self.total_weight_grad[j], (np.sqrt(np.abs(self.total_sq_weight_grad[j])) + 1e-8))

			if normalize:
				norm = la.norm(update_val)
				if norm != 0:
					update_val = update_val / norm
			curr_weight_update_vals.append(update_val)
			update = tf.assign(self.weights[j], self.weights[j] + update_val)
			self.sess.run(update)

		# Update biases
		for j in range(len(self.biases)):
			if update_method == 'sgd':
				update_val = step_size * curr_bias_grad[j]

			elif update_method == 'momentum':
				if self.prev_bias_grad is None:
					update_val = step_size * curr_bias_grad[j]
				else:
					update_val = momentum * self.prev_bias_grad[j] +  step_size * curr_bias_grad[j]

			elif update_method == 'lbfgs':
				if self.prev_bias_inverse_hess is None:
					curr_inverse_hess = np.eye(curr_bias_grad[j].shape[0])
					update_val = curr_bias_grad[j]
				else: 
					update_val, curr_inverse_hess = \
						self.bfgs_update(self.prev_inverse_hess[j], self.prev_update_val[j], self.prev_bias_grad[j], update_val)
				update_val = update_val * step_size 
				curr_bias_inverse_hess.append(curr_inverse_hess)

			elif update_method == 'adagrad':
				if self.total_bias_grad is None:
					self.total_bias_grad = curr_bias_grad
				else:
					self.total_bias_grad[j] += np.square(curr_bias_grad[j])
				update_val = step_size * curr_bias_grad[j] / (np.sqrt(np.abs(self.total_bias_grad[j])) + 1e-8)

			elif update_method == 'rmsprop':
				decay = 0.99
				if self.total_bias_grad is None:
					self.total_bias_grad = curr_bias_grad
				else:
					self.total_bias_grad[j] = decay * self.total_bias_grad[j] + (1 - decay) * np.square(curr_bias_grad[j])
				update_val = step_size * curr_bias_grad[j] / (np.sqrt(np.abs(self.total_bias_grad[j])) + 1e-8)

			elif update_method == 'adam':
				beta1, beta2 = 0.9, 0.999
				if self.total_bias_grad is None:
					self.total_bias_grad = curr_bias_grad
					self.total_sq_bias_grad = np.square(curr_bias_grad)
				else:
					self.total_bias_grad[j] = beta1 * self.total_bias_grad[j] + (1 - beta1) * curr_bias_grad[j]
					self.total_sq_bias_grad[j] = beta2 * self.total_sq_bias_grad[j] + (1 - beta2) * np.sqrt(np.abs(self.total_bias_grad[j]))
				update_val = np.divide(step_size * self.total_bias_grad[j], (np.sqrt(np.abs(self.total_sq_bias_grad[j])) + 1e-8))

			if normalize:
				norm = la.norm(update_val)
				if norm != 0:
					update_val = update_val / norm
			curr_bias_update_vals.append(update_val)
			update = tf.assign(self.biases[j], self.biases[j] + update_val)
			self.sess.run(update)

		self.prev_weight_grad = curr_weight_grad
		self.prev_bias_grad = curr_bias_grad
		self.prev_weight_update_vals = curr_weight_update_vals
		self.prev_bias_update_vals = curr_weight_update_vals


	def get_action(self, state):
		"""
		Returns action based on input state.

		Input:
		state: array-like
			Input state.

		Output:
		action: array-like
			Predicted action.
		"""
		state = state.T
		curr_output_mean = self.sess.run(self.output_mean, feed_dict={self.input_state: state})
		action = self.meanstd_sample(curr_output_mean)
		return action

	

