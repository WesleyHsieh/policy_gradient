import numpy as np
import numpy.linalg as la
import policy_gradient
import matplotlib.pyplot as plt

class Testbed:
	"""
	Simple testbed for policy gradient.
	"""
	def __init__(self, A, B):
		"""
		Initializes dynamics of environment.

		Parameters:
		A: array-like
			Transition matrix for states.
		B: array-like
			Transition matrix for actions.
		"""
		self.A = A
		self.B = B
		self.goal_state = np.array([10,10]).reshape(2,1)
		self.learner = policy_gradient.Policy_Gradient(goal_state=self.goal_state, net_dims=[2,2,2,2])

	def calculate_next_state(self, state, action):
		"""
		Calculates next state using given state-action pair.

		Equation: x_{t+1} = Ax + Bu + w

		Parameters:
		state: array-like
			Input state.
		action: array-like
			Action taken from given state.
		"""
		next_state = np.dot(self.A, state) + np.dot(self.B, action) #+ np.random.normal(loc=0.0, scale=1.0, size=state.shape)
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
		return -la.norm(s - self.goal_state)

	def run_test(self, iters=100, iter_len=100, traj_len=10):
		"""
		Runs learner over example environment 
		and returns results.

		Parameters:
		iter_len: int
			Number of iterations.
		traj_len: int
			Length of each sample trajectory.
		"""
		
		mean_rewards = []
		for i in range(iters):
			traj_states = []
			traj_actions = []
			rewards = []
			for j in range(iter_len):
				states = []
				actions = []
				curr_rewards = []
				curr_state = np.zeros((self.A.shape[0], 1))

				# Rolls out single trajectory
				for k in range(traj_len):
					# Get action from learner
					curr_action = self.learner.get_action(curr_state)

					# Update values
					states.append(curr_state)
					curr_rewards.append(self.reward_function(curr_state, curr_action))
					actions.append(curr_action)

					# Update state
					curr_state = self.calculate_next_state(curr_state, curr_action)

				# Append trajectory/rewards
				traj_states.append(states)
				traj_actions.append(actions)
				rewards.append(curr_rewards)

			# Apply policy gradient iteration
			self.learner.gradient_update(np.array(traj_states), np.array(traj_actions), np.array(rewards))
			mean_rewards.append(np.mean([reward_list[-1] for reward_list in rewards]))

		mean_rewards = -np.array(mean_rewards)
		print "Mean Ending Distance: \n {}".format(mean_rewards)
		# Display results
		plt.figure()
		plt.plot(range(len(mean_rewards)), mean_rewards)
		plt.xlabel('Number of Iterations')
		plt.ylabel('Mean Ending Distance from Goal')
		plt.title('Policy Gradient Learning')
		plt.savefig("policy_gradient_{}_iters_{}_iterlen.pdf".format(iters, iter_len))

	def run_step(self, action):
		"""
		Performs a single trajectory run.
		"""
		next_state = self.calculate_next_state(action)
		reward = self.reward_function(self.state, action)
		return next_state, reward, terminated

if __name__ == '__main__':
	A = np.diag(np.ones(2))
	B = np.diag(np.ones(2))
	testbed = Testbed(A, B)
	testbed.run_test()