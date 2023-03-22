from setuptools import setup

"""
Version legend:
a.b.c.d

a: major release
b: new functionality added
c: new feature to existing functionality
d: bug fixes
"""

setup(
    name = 'kshell-py',
    version = '0.0.0.0',
    description = 'Python implementation of the nuclear shell model solver, KSHELL.',
    url = 'https://github.com/GaffaSnobb/kshell-py',
    author = 'Jon Kristian Dahl',
    author_email = 'jonkd@uio.no',
    packages = ['kshell_py', 'test'],
    install_requires = ['numpy'],

    classifiers = [
        # 'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Operating System :: MacOS',
        'Operating System :: Unix',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Physics',
    ],
)
