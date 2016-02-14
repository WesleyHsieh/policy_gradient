import numpy as np
import numpy.linalg as la
import collections
import IPython
import tensorflow as tf
from utils import *

class PolicyGradient(Utils):

	""" 
	Calculates policy gradient
	for given input state/actions.

	Users should primarily be calling main
	PolicyGradient class methods.
	"""

	def __init__(self, net_dims, output_function=None):
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
		self.prev_weight_update = self.prev_bias_update = None
		self.init_neural_net(net_dims, output_function)

	def train_agent(self, dynamics_func, reward_func, initial_state, num_iters, batch_size, traj_len, \
			step_size=0.1, momentum=0.5, normalize=True):
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
					step_size, momentum, normalize)
			mean_rewards.append(np.mean([reward_list[-1] for reward_list in rewards]))
		return np.array(mean_rewards)

	def gradient_update(self, traj_states, traj_actions, rewards, step_size=0.1, momentum=0.5, normalize=True):
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
		# Calculate updates and create update pairs
		weight_update = 0
		bias_update = 0
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
			for j in range(curr_q_val_list.shape[0]):
				# Extract current state, action, q-value
				curr_state = curr_traj_states[j].T
				curr_action = curr_traj_actions[j].T
				curr_q_val = curr_q_val_list[j]

				# Calculate gradients
				# print "curr_state", curr_state
				# print "curr_action", curr_action
				# print "curr_q_val", curr_q_val
				weight_update_vals = self.sess.run(self.weight_grads, \
					feed_dict={self.input_state: curr_state, self.observed_action: curr_action, self.q_val: curr_q_val})
				bias_update_vals = self.sess.run(self.bias_grads, \
					feed_dict={self.input_state: curr_state, self.observed_action: curr_action, self.q_val: curr_q_val}) 
				weight_update += np.array(weight_update_vals) / np.float(iters)
				bias_update += np.array(bias_update_vals) / np.float(iters)
		
		# Update weights
		for j in range(len(self.weights)):
			# Normalize gradient
			update_val = weight_update[j]
			if normalize:
				norm = la.norm(weight_update[j])
				if norm != 0:
					update_val = weight_update[j] / la.norm(weight_update[j])
			if momentum != 0 and self.prev_weight_update is not None:
				update_val += momentum * self.prev_weight_update[j]
			update = tf.assign(self.weights[j], self.weights[j] + step_size * update_val)
			self.sess.run(update)

		# Update biases
		for j in range(len(self.biases)):
			# Normalize gradient
			update_val = bias_update[j]
			if normalize:
				norm = la.norm(bias_update[j])
				if norm != 0:
					update_val = bias_update[j] / la.norm(bias_update[j])
			if momentum != 0 and self.prev_bias_update is not None:
				update_val += momentum * self.prev_bias_update[j]
			update = tf.assign(self.biases[j], self.biases[j] + step_size * update_val)
			self.sess.run(update)

		self.prev_weight_update = weight_update
		self.prev_bias_update = bias_update
		return None

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

	

