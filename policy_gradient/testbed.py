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
	def __init__(self, A=None, B=None, goal_state=None, net_dims=None):
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
		self.net_dims = net_dims if net_dims is not None else [2,6,4,3,2]
		self.learner = policy_gradient.PolicyGradient(net_dims=self.net_dims, output_function='tanh')

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
		plt.savefig("policy_gradient_{}_numiters_{}_batchsize.pdf".format(num_iters, batch_size))

if __name__ == '__main__':
	start_time = time.clock()
	testbed = Testbed(goal_state = np.array([10,10]))
	# print "Running Test Suite."
	# call(["nosetests", "-v", "testbed.py"])
	print "Running Performance Test."
	num_iters = 50
	batch_size = 50
	testbed.performance_check(num_iters, batch_size)
	end_time = time.clock()
	total_time = end_time - start_time
	print "{} seconds elapsed for {} iterations of {} rollouts each.".format(total_time, num_iters, batch_size)