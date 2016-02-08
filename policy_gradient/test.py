class A:
	def __init__(self):
		self.x = 1
	def test1(self):
		return self.x
class B:
	def __init__(self):
		self.y = 2
	def test1(self):
		return self.y
	def test2(self, func):
		return func()

a = A()
b = B()
print b.test2(a.test1)