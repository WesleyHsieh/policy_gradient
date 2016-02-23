import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import policy_gradient
import testclass
from subprocess import call
import time

class Testbed(testclass.TestClass):
	"""
	Simple testbed for policy gradient.
	"""
	def __init__(self, A=None, B=None, goal_state=None, net_dims=None, obstacles=None):
		"""
		Initializes dynamics of environment.

		Parameters:
		A: array-like
			Transition matrix for states.
		B: array-like
			Transition matrix for actions.
		"""
		self.A = A if A is not None else np.diag(np.ones(2)) 
		self.B = B if B is not None else np.diag(np.ones(2))
		self.goal_state = goal_state if goal_state is not None else np.array([10,10]).reshape(2,1)
		self.net_dims = net_dims if net_dims is not None else [2,4,3,2]
		self.learner = policy_gradient.PolicyGradient(net_dims=self.net_dims, output_function='tanh')
		self.obstacles = obstacles

	def calculate_next_state(self, state, action):
		"""
		Calculates next state using given state-action pair.

		Equation: x_{t+1} = Ax_{t} + Bu + w

		Parameters:
		state: array-like
			Input state.
		action: array-like
			Action taken from given state.
		"""
		next_state = np.dot(self.A, state) + np.dot(self.B, action) + np.random.normal(loc=0.0, scale=1.0, size=state.shape)
		return next_state

	def reward_function(self, s, a):
		"""
		Calculates rewards for a state-action pair.

		Parameters:
		s: state object
			Input state.

		a: array-like
			Input action.

		Output:
		r: float
			Reward, l2 norm between current and goal state.
		"""
		curr_reward = la.norm(s - self.goal_state)
		next_state = self.calculate_next_state(s, a)
		next_reward = la.norm(next_state - self.goal_state)
		return curr_reward - next_reward

	def run_step(self, action):
		"""
		Performs a single trajectory run.
		"""
		next_state = self.calculate_next_state(action)
		reward = self.reward_function(self.state, action)
		return next_state, reward, terminated

	def performance_check(self, num_iters=100, batch_size=100, traj_len=10):
		"""
		Runs learner over example environment 
		and returns results.

		Parameters:
		num_iters: int
			Number of iterations to run gradient updates.
		batch_size: int
			Number of trajectories to run in a single iteration.
		traj_len: int
			Number of state-action pairs in a trajectory.
		"""
		initial_state = np.zeros((self.A.shape[0], 1))
		mean_rewards, ending_states = self.learner.train_agent(dynamics_func=self.calculate_next_state, reward_func=self.reward_function, \
			initial_state=initial_state, num_iters=num_iters, batch_size=batch_size, traj_len=traj_len)
		mean_rewards = np.array(mean_rewards)
		mean_ending_states = np.mean(ending_states, axis=1)
		mean_ending_distances = [la.norm(s - self.goal_state) for s in mean_ending_states]
		print "Mean Ending Distance: \n {}".format(np.around(mean_ending_distances, decimals=3))
		# Display results
		plt.figure()
		plt.plot(range(len(mean_ending_distances)), mean_ending_distances)
		plt.xlabel('Number of Iterations')
		plt.ylabel('Mean Ending Distance from Goal')
		plt.title('Policy Gradient Learning')
		plt.savefig("figures/policy_gradient_{}_numiters_{}_batchsize.pdf".format(num_iters, batch_size))

	def cross_validate(self, step_size, momentum, net_dims, q_net_dims, update_method, seed_state, num_iters=50, batch_size=50, traj_len=20, seed=0):

		learner = policy_gradient.PolicyGradient(net_dims=net_dims, q_net_dims=q_net_dims, output_function='tanh', seed=seed, seed_state=seed_state)
		initial_state = np.zeros((self.A.shape[0], 1))
		mean_rewards, ending_states = learner.train_agent(dynamics_func=self.calculate_next_state, reward_func=self.reward_function, \
			update_method=update_method, initial_state=initial_state, num_iters=num_iters, batch_size=batch_size, traj_len=traj_len, \
			step_size=step_size, momentum=momentum, normalize=False)
		mean_ending_states = np.mean(ending_states, axis=1)
		mean_ending_distances = [la.norm(s - self.goal_state) for s in mean_ending_states]
		return mean_ending_distances


# Cross-Validate step size, momentum, neural net params, learned q function, lbfgs
# Plot how performance scales with num_iters, batch_size, goal state distance
if __name__ == '__main__':
	np.random.seed(0)
	st0 = np.random.get_state()
	testbed = Testbed(goal_state = np.array([20,20]))
	print "Running Performance Test."
	update_method_list = ['sgd', 'momentum', 'lbfgs', 'adagrad', 'rmsprop', 'adam']
	seeds = [0]
	num_iters = 50
	batch_size = 50
	step_size = 0.1
	momentum = 0.5 
	net_dims = [2,4,3,2]
	q_net_dims = None
	plt.figure()
	plt.xlabel('Number of Iterations')
	plt.ylabel('Mean Ending Distance from Goal')
	plt.title('Policy Gradient Learning')
	for update_method in update_method_list:
		distance_list = []
		for i in range(5):
			mean_ending_distances = testbed.cross_validate(step_size, momentum, net_dims, q_net_dims, update_method, st0, num_iters, batch_size, seed=0)
			distance_list.append(mean_ending_distances)
		distance_list = np.mean(np.array(distance_list), axis=0)
		plt.plot(range(len(distance_list)), distance_list, label='update_method={}'.format(update_method))

	plt.legend()
	plt.savefig("figures/policy_gradient_iters.pdf")
