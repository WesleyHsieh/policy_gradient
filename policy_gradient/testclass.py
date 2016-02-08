import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from policy_gradient import *
import IPython

class TestClass:
	def __init__(self):
		self.goal_state = np.array([10,10]).reshape(2,1)

	def test_meanstd_sample(self):
		pol_grad = PolicyGradient(net_dims=[2,4,3,1])
		input_mean = np.array([10, 20, -5])
		output_sample = pol_grad.meanstd_sample(mean=input_mean, std=0)
		assert np.array_equal(input_mean, output_sample)

	def test_setup_net(self):
		input_net_dims = [2,4,3,1]
		pol_grad = PolicyGradient(net_dims=[2,4,3,1])
		assert len(pol_grad.layers) == 4
		assert np.array_equal(pol_grad.layers[0].get_shape().as_list(), [None, 2])
		assert np.array_equal(pol_grad.layers[1].get_shape().as_list(), [None, 4])
		assert np.array_equal(pol_grad.layers[2].get_shape().as_list(), [None, 3])
		assert np.array_equal(pol_grad.layers[3].get_shape().as_list(), [None, 1])
		assert len(pol_grad.weights) == 3
		assert np.array_equal(pol_grad.weights[0].get_shape().as_list(), [2,4])
		assert np.array_equal(pol_grad.weights[1].get_shape().as_list(), [4,3])
		assert np.array_equal(pol_grad.weights[2].get_shape().as_list(), [3,1])
		assert len(pol_grad.biases) == 3
		assert np.array_equal(pol_grad.biases[0].get_shape().as_list(), [4])
		assert np.array_equal(pol_grad.biases[1].get_shape().as_list(), [3])
		assert np.array_equal(pol_grad.biases[2].get_shape().as_list(), [1])
		assert np.array_equal(pol_grad.input_state.get_shape().as_list(), [None, 2])

	def debug_mode(self):
		test_pol_grad = PolicyGradient(net_dims=[2,2,2,2])
		IPython.embed()