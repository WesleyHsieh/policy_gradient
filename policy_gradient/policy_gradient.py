import numpy as np
import numpy.linalg as la
import collections
import IPython
import tensorflow as tf

class Policy_Gradient:

	""" 
	Calculates policy gradient
	for given input state/actions.
	"""

	def __init__(self, goal_state, net_dims):
		self.goal_state = goal_state
		self.prev_weight_update = self.prev_bias_update = None
		self.init_neural_net(net_dims)

	def setup_net(self, net_dims):
		"""
		Initializes TensorFlow neural net 
		with input net dimensions.

		Parameters:
		net_dims: array-like
			List of dimensions for layers of net.

		Output:
		layers: array-like
			List of TensorFlow nodes corresponding to 
			layers of net.

		weights: array-like
			Net weights.

		biases: array-like
			Net biases.

		input_state: TensorFlow node
			Placeholder for input state.
		"""
		# Initialize placeholders for input state, weights, biases
		input_state = tf.placeholder("float", [None, net_dims[0]])
		weights = [tf.Variable(tf.random_normal([net_dims[i], net_dims[i+1]])) for i in range(len(net_dims) - 1)]
		biases = [tf.Variable(tf.random_normal([net_dims[i]])) for i in range(1, len(net_dims))]
		assert len(weights) == len(biases)
		assert len(net_dims) == len(weights) + 1 

		# Iteratively construct layers based on previous layer
		layers = [input_state]
		for i in range(len(weights)):
			prev_layer, prev_weights, prev_biases = layers[i], weights[i], biases[i]
			# Apply relu to all but last layer
			if i == len(weights) - 1:
				layers.append(tf.add(tf.matmul(prev_layer, prev_weights), prev_biases))
			else:
				layers.append(tf.nn.relu(tf.add(tf.matmul(prev_layer, prev_weights), prev_biases)))
		assert len(layers) == len(net_dims)
		return layers, weights, biases, input_state

	def estimate_q(self, states, actions, rewards, net_dims=None):
		"""
		Estimates the q-values for a trajectory
		based on the intermediate rewards.

		Parameters:
		states: array-like
			List of list of states.

		actions: array-like
			List of list of actions.

		rewards: array-like
			List of list of rewards.

		net_dims: array-like
			List of dimensions for layers of net.
			Defaults to three layer net, with dimensions 
			x, 2x, 1.5x, 1 respectively where x is the size
			of the trajectory.

		Output: 
		q: array-like
			Estimated q-values.
		"""
		q_temp = []
		for i in range(rewards.shape[0]):
			q_list = []
			for j in range(rewards.shape[1]):
				q_list.append(np.sum(rewards[i,j:]))
			q_temp.append(q_list)
		q_temp = np.array(q_temp)
		return q_temp

		# traj_flattened = np.concatenate((states, actions), axis=2)
		# traj_reshaped = traj_flattened.reshape(traj_flattened.shape[0] * traj_flattened.shape[1], traj_flattened.shape[2])
		# if net_dims is None:
		# 	state_shape = traj_reshaped.shape[1]
		# 	net_dims = [state_shape, 2 * state_shape, np.round(1.5 * state_shape), 1]

		# # Setup net
		# layers, _, _ = self.setup_net(net_dims)

		# # Preprocess q
		# q = []
		# for i in range(rewards.shape[0]):
		# 	curr_reward_list = rewards[i]
		# 	curr_q_list = []
		# 	curr_q_val = 0
		# 	# Compute q-values backwards
		# 	for j in range(rewards.shape[1])[::-1]:
		# 		curr_q_val += curr_reward_list[j]
		# 		curr_q_list.append(curr_q_val)
		# 	q.append(curr_q_list[::-1])
		# q = np.array(q)
		# q_reshaped = q.reshape(q.shape[0] * q.shape[1])

		# input_state = layers[0]
		# output_state = layers[-1]
		# # Update net (TODO)
		# outputs = self.sess.run(output_state, feed_dict={input_state: traj_reshaped})
		# return np.array(q)


	def gradient_update(self, traj_states, traj_actions, rewards, step_size=0.01, momentum=0.0, normalize=True):
		"""
		Estimates and applies gradient update according to a policy.

		Parameters:
		traj_states: array-like
			List of states.

		traj_actions: array-like
			List of actions.

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
			if momentum:
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
			if momentum:
				update_val += momentum * self.prev_bias_update[j]
			update = tf.assign(self.biases[j], self.biases[j] + step_size * update_val)
			self.sess.run(update)

		self.prev_weight_update = weight_update
		self.prev_bias_update = bias_update
		return None


	def init_neural_net(self, net_dims):
		"""
		Sets up neural network for policy gradient.

		Input:
		net_dims: array-like
			List of dimensions for layers of net
.		"""
		# tf Graph input
		n_inputs = net_dims[0]
		n_outputs = net_dims[-1]
		#self.input_state = tf.placeholder("float", [None, n_inputs])
		self.observed_action = tf.placeholder("float", [None, n_outputs])
		self.q_val = tf.placeholder("float")

		# Setup neural net
		layers, self.weights, self.biases, self.input_state = self.setup_net(net_dims)
		self.output_mean = layers[-1]

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
		action = meanstd_sample(curr_output_mean)
		return action

def meanstd_sample(mean, std=1.0):
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
	sample_action = mean + std * np.random.randn(*mean.shape)
	return sample_action.T
