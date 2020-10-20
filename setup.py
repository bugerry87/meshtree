from setuptools import setup
import meshtree

setup(
    name='meshtree',
    version=meshtree.__version__,
    description=meshtree.__description__,
    long_description=open('README.md').read(),
    url='https://github.com/bugerry87/meshtree',
    author='Gerald Baulig',
    author_email='gerald.baulig@gmail.com',
    license='MIT',
    classifiers=[
        'Development Status :: Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
		'Operating System :: OS Independent'
    ],
    keywords='3d mesh nearest-neighbor kdtree',
    packages=['meshtree'],
	scripts=['meshtree_demo.py'],
	install_requires=['numpy']
)