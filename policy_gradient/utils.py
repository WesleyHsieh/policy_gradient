import numpy as np
import numpy.linalg as la
import collections
import IPython
import tensorflow as tf

class Utils:
	"""
	Utility class containing miscellaneous
	functions.

	Users should primarily be calling main
	PolicyGradient class methods.
	"""
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
			List of net weights.

		biases: array-like
			List of net biases.

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

	def estimate_q(self, states, actions, rewards, net_dims=None, net=False):
		"""
		Estimates the q-values for a trajectory
		based on the intermediate rewards.

		Parameters:
		states: array-like
			List of list of input states.

		actions: array-like
			List of list of input actions.

		rewards: array-like
			List of list of input rewards.

		net_dims: array-like
			List of dimensions for layers of net.
			Defaults to three layer net, with dimensions 
			x, 2x, 1.5x, 1 respectively where x is the length
			of the trajectory.

		Output: 
		q: array-like
			Estimated q-values.
		"""
		q = []
		for i in range(rewards.shape[0]):
			curr_reward_list = rewards[i]
			curr_q_list = []
			curr_q_val = 0
			# Compute q-values backwards
			for j in range(rewards.shape[1])[::-1]:
				curr_q_val += curr_reward_list[j]
				curr_q_list.append(curr_q_val)
			q.append(curr_q_list[::-1])
		q = np.array(q)

		if net:
			traj_flattened = np.concatenate([states, actions], axis=2)
			traj_reshaped = traj_flattened.reshape(traj_flattened.shape[0] * \
					traj_flattened.shape[1], traj_flattened.shape[2])
			if net_dims is None:
				state_shape = int(traj_reshaped.shape[1])
				net_dims = [state_shape, 2 * state_shape, int(np.round(1.5 * state_shape)), 1]

			# Setup net
			layers, _, _, input_state = self.setup_net(net_dims)

			q_processed = q.reshape(q.shape[0] * q.shape[1], 1)
			q_var = tf.placeholder("float", [None, 1])
			input_state = layers[0]
			output_state = layers[-1]
			
			l2_loss = tf.nn.l2_normalize(tf.sub(q_var, output_state), dim=0)
			train_step = tf.train.AdamOptimizer().minimize(l2_loss)

			init = tf.initialize_all_variables()
			with tf.Session() as sess:
				sess.run(init)
				train_step.run(feed_dict={input_state: traj_reshaped, q_var: q_processed})
				output_q = sess.run(output_state, feed_dict={input_state: traj_reshaped})
			reshaped_output_q = output_q.reshape(rewards.shape)

			return reshaped_output_q
		else:
			return q

	def bfgs_update(self, prev_inverse_hess, update_x, prev_grad, curr_grad):
		s = update_x
		y = curr_grad - prev_grad
		b = prev_inverse_hess

		t1 = np.div(np.dot(s, y.T), np.dot(y.T, s))
		t2 = np.div(np.dot(y, s.T), np.dot(y.T, s))
		t3 = np.div(np.dot(s, s.T), np.dot(y.T, s))

		t1 = np.eye(t1.shape) - t1
		t2 = np.eye(t2.shape) - t2
		curr_inverse_hess = np.dot(np.dot(t1, b), t2) + t3

		update_val = -np.dot(curr_inverse_hess, curr_grad)
		return update_val, curr_inverse_hess


	def init_action_neural_net(self, net_dims, output_function=None):
		"""
		Sets up neural network for policy gradient.

		Input:
		net_dims: array-like
			List of dimensions for layers of net.
		output_function: string
			Non-linearity function applied to output of
			neural network. 
			Options are: 'tanh', 'sigmoid', 'relu'.
		"""
		assert output_function in [None, 'tanh', 'sigmoid', 'relu', 'softmax']

		n_inputs = net_dims[0]
		n_outputs = net_dims[-1]
		self.observed_action = tf.placeholder("float", [None, n_outputs])
		self.q_val = tf.placeholder("float", [None, 1])

		# Setup neural net
		self.layers, self.weights, self.biases, self.input_state = self.setup_net(net_dims)
		self.output_mean = self.layers[-1]
		if output_function == "tanh":
			self.output_mean = tf.nn.tanh(self.output_mean)
		elif output_function == "sigmoid":
			self.output_mean = tf.nn.sigmoid(self.output_mean)
		elif output_function == "relu":
			self.output_mean = tf.nn.relu(self.output_mean)
		elif output_function == 'softmax':
			self.output_mean = tf.nn.softmax(self.output_mean)

		 # Output probability layer
		diff = tf.sub(self.output_mean, self.observed_action)
		norm_diff = tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(diff), reduction_indices=1)), 1)
		log_prob_output = tf.mul(tf.constant(-0.5), tf.square(norm_diff))
		prob_q = tf.mul(log_prob_output, self.q_val)
		self.prob_q = prob_q
		# Gradients
		self.weight_grads = tf.gradients(prob_q, self.weights)
		self.bias_grads = tf.gradients(prob_q, self.biases)

		# Initialize variables, session
		init = tf.initialize_all_variables()
		self.sess = tf.Session()
		self.sess.run(init)


	def meanstd_sample(self, mean, std=1.0):
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
