import numpy as np
import pylab

values = np.random.rand(2, 4, 3)
print(values)

def modify(values):
	values = values * 1.1
	print("modified", values)
	return values


class plotter:
	def __init__(self, values):
		self.values = values
		self.fig = pylab.figure()
		pylab.gray()
		self.ax = self.fig.add_subplot(1, 1, 1, projection="3d")
		self.draw()
		self.fig.canvas.mpl_connect('key_press_event',self.key)

	def draw(self):
		x = self.values[:, 0:1]
		y = self.values[:, 1:2]
		z = self.values[:, 2:3]

		self.ax.scatter(x, z, y, marker=".", s=1)
		pylab.show()

	def key(self, event):
		if event.key=='right':
			self.values = modify()
		elif event.key == 'left':
			self.values = modify()

		self.draw()
		self.fig.canvas.draw()


plot = plotter(values)
plot.draw