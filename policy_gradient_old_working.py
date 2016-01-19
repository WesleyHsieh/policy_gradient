import numpy as np
import numpy.linalg as la
import collections
import cgt
from cgt import nn
import IPython
import tensorflow as tf

class Policy_Gradient:

	""" 
	Calculates policy gradient
	for given input state/actions.
	"""

	def __init__(self, goal_state=None):
		self.goal_state = goal_state
		self.init_neural_net()


	def estimate_q(self, traj, rewards):
		"""
		Estimates the q-values for a trajectory
		based on the intermediate rewards.

		Parameters:
		traj: array-like
			List of state-action pairs.

		rewards: array-like
			List of rewards.

		Output: 
		q: array-like
			Estimated q-values.

		q_dict: dict
			Dictionary of q-value, indexed by state
		"""
		q = []
		for i in range(len(rewards)):
			q.append(np.sum(rewards[i:]))
		return np.array(q)


	def gradient_update(self, traj, rewards, step_size=0.01):
		"""
		Estimates and applies gradient update according to a policy.

		Parameters:
		traj: array-like
			List of state-action pairs.

		rewards: array-like
			List of rewards.

		stepsizeval: float
			Step size.

		Output: 
			None.
		"""
		

		# Calculate updates and create update pairs
		weight_update = 0
		bias_update = 0
		iters = traj.shape[0]
		# Update for each example
		for i in range(iters):
			# Estimate q-values and extract gradients
			curr_traj, curr_rewards = traj[i], rewards[i]
			q = self.estimate_q(curr_traj, curr_rewards)
			assert q.shape[0] == curr_traj.shape[0]
			for j in range(q.shape[0]):
				# Extract current state, action, q-value
				curr_state = curr_traj[j][0].T
				curr_action = curr_traj[j][1].T
				curr_q_val = q[j]

				#print "q", curr_q_val
				# Calculate gradients
				weight_update_vals = self.sess.run(self.weight_grads, \
					feed_dict={self.input_state: curr_state, self.observed_action: curr_action, self.q_val: curr_q_val})
				bias_update_vals = self.sess.run(self.bias_grads, \
					feed_dict={self.input_state: curr_state, self.observed_action: curr_action, self.q_val: curr_q_val}) 

				weight_update += np.array(weight_update_vals) / np.float(iters)
				bias_update += np.array(bias_update_vals) / np.float(iters)
		
		# Update weights
		for j in range(len(self.weights)):
			# Normalize gradient
			norm = la.norm(weight_update[j])
			if norm != 0:
				update_val = weight_update[j] / la.norm(weight_update[j])
				#print "update weights {}:, {}".format(j, update_val)
				update = tf.assign(self.weights[j], self.weights[j] + step_size * update_val)
				self.sess.run(update)

		# Update biases
		for j in range(len(self.biases)):
			# Normalize gradient
			norm = la.norm(bias_update[j])
			if norm != 0:
				update_val = bias_update[j] / la.norm(bias_update[j])
				#print "update weights {}:, {}".format(j, update_val)
				update = tf.assign(self.biases[j], self.biases[j] + step_size * update_val)
				self.sess.run(update)

		return None


	def init_neural_net(self, n_inputs=2, n_outputs=2, n_hidden_1=2, n_hidden_2=2):
		"""
		Sets up neural network for policy gradient.

		Input:
		n_inputs: int
			Number of inputs to neural net.

		n_outputs: int
			Number of outputs from neural net.

		n_hidden_1: int
			Number of units in first hidden layer.

		n_hidden_2: int
			Number of units in second hidden layer.
		"""
		# tf Graph input
		self.input_state = tf.placeholder("float", [None, n_inputs])
		self.observed_action = tf.placeholder("float", [None, n_outputs])
		self.q_val = tf.placeholder("float")


		# Store layers weight & bias
		self.weights = [tf.Variable(tf.random_normal([n_inputs, n_hidden_1])),
				   tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
				   tf.Variable(tf.random_normal([n_hidden_2, n_outputs]))]
		self.biases = [tf.Variable(tf.random_normal([n_hidden_1])),
				  tf.Variable(tf.random_normal([n_hidden_2])),
				  tf.Variable(tf.random_normal([n_outputs]))]

		# Construct model
		layer_1 = tf.nn.relu(tf.add(tf.matmul(self.input_state, self.weights[0]), self.biases[0])) #Hidden layer with RELU activation
		layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, self.weights[1]), self.biases[1])) #Hidden layer with RELU activation
		self.output_mean = tf.matmul(layer_2, self.weights[2]) + self.biases[2]

		 # Output probability layer
		log_prob_output = tf.mul(tf.constant(-0.5), tf.square(tf.global_norm([self.output_mean - self.observed_action])))
		prob_q = tf.mul(self.q_val, log_prob_output)

		# Gradients
		self.weight_grads = tf.gradients(prob_q, self.weights)
		self.bias_grads = tf.gradients(prob_q, self.biases)

		# Initialize variables, session
		init = tf.initialize_all_variables()
		self.sess = tf.Session()
		self.sess.run(init)

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
		action = meanstd_sample(curr_output_mean, std=1)
		return action

def meanstd_sample(mean, std=0.001):
	"""
	Samples an action based on
	the input probability distribution.

	Input:
	mean: array-like
		Input mean of action distribution.

	std: float
		Standard deviation of action distribution.

	Output:
	sample_action: array-like
		Action sampled from given distribution.
	"""
	#print "mean", mean
	sample_action = mean + std * np.random.randn(*mean.shape)
	#print "sample_action", sample_action
	return sample_action.T