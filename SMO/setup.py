from distutils.core import setup, Extension
import numpy.distutils.misc_util

setup(
	name = "linear-smo",
	version = "0.0.1",
	author = "Dominik Stanojevic",
	author_email = "dominik.stanojevic@fer.hr",
	description = ("Small C-extension for linear svm using dual coordinate ascent algorithm."),
	keywords = "smo svm linear dual coordinate ascent",
    ext_modules=[Extension("smo", ["_smo.c", "smo.c"])],
    include_dirs=numpy.distutils.misc_util.get_numpy_include_dirs(),
)
