#!/usr/bin/env python3

## Build In
from argparse import ArgumentParser
from time import time

## Installed
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import meshtree


def init_main_args(parents=[]):
	"""
	Initialize an ArgumentParser for this module.
	
	Args:
		parents: A list of ArgumentParsers of other scripts, if there are any.
		
	Returns:
		parser: The ArgumentParsers.
	"""
	parser = ArgumentParser(
		description=__description__,
		conflict_handler='resolve',
		parents=parents
		)
	
	parser.add_argument(
		'--vertices', '-X',
		metavar='PATH',
		help='A NPY-file of verticies'
		)
	
	parser.add_argument(
		'--indices', '-T',
		metavar='PATH',
		help='A NPY-file of indices indecating vertice connecting'
		)
	
	parser.add_argument(
		'--query', '-P',
		metavar='PATH',
		help='A NPY-file of query points'
		)
	
	parser.add_argument(
		'--demo',
		type=int,
		metavar='INT',
		default=1000,
		help='Run a demo with N random points'
		)
	
	return parser


def time_delta(start=None):
	"""
	Captures time delta from last call.
	
	Args:
		start: Set start time.
	
	Yields:
		delta: Past time in seconds.
	"""
	if not start:
		start = time()
	
	while True:
		curr = time()
		delta = curr - start
		start = curr
		yield delta


def callback(tree):
	"""
	"""
	last = 0
	while last <= 50:
		curr = int(tree.done.mean() * 50)
		dif = curr - last
		if curr > last:
			print('#' * dif, end='', flush=True)
		last = curr
		yield


def main(args):
	"""
	"""
	np.random.seed(0)
	X = np.load(args.vertices) if args.vertices else np.random.randn(args.demo*3, 3)
	Xi = np.load(args.indices) if args.indices else np.arange(len(X)).reshape(-1, 3)
	P = np.load(args.query) if args.query else np.random.randn(args.demo, 3)
	
	print("MeshTree")
	print("Model size:", X.shape)
	print("Query size:", P.shape)
	
	delta = time_delta(time())
	tree = MeshTree(
		X, Xi,
		callback=callback,
		**args.__dict__
		)
	
	print("\n0%					  |50%					 |100%")
	dist, mp, nn = tree.query(P)
	
	print("\nQuery time:", next(delta))
	print("Mean loss:", np.sqrt(dist).mean())
	
	fig = plt.figure()
	ax = fig.add_subplot((111), projection='3d')
	mp -= P
	typ = np.sum(nn == -1, axis=-1)
	point = typ == 2
	line = typ == 1
	face = typ == 0
	
	poly = Poly3DCollection(X)
	poly.set_alpha(0.5)
	poly.set_edgecolor('b')
	ax.add_collection3d(poly)
	ax.scatter(*P.T, color='r')
	ax.quiver(*P[point].T, *mp[point].T, color='g')
	ax.quiver(*P[line].T, *mp[line].T, color='y')
	ax.quiver(*P[face].T, *mp[face].T, color='b')
	
	ax.set_xlim3d(-3, 3)
	ax.set_ylim3d(-3, 3)
	ax.set_zlim3d(-2.4, 2.4)
	plt.show()
	print(tree)


if __name__ == '__main__':
	main_args = init_main_args()
	meshtree.init_meshtree_args([main_args])
	args, _ = main_args.parse_known_args()
	main(args)