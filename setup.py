from setuptools import setup, find_packages
import os

with open("README.md", "r") as fh:
  long_description = fh.read()

def getPackageInfo():
  about = {}
  with open(os.path.join("lightem", "version.py"), "r") as f:
    exec(f.read(), about)
  return about

about = getPackageInfo()

# pede permissão de adm pra apagar as pastas build, dist e lightem.egg-info
os.system("rmdir /S /Q build")
os.system("rmdir /S /Q dist")
os.system("rmdir /S /Q lightem.egg-info")

setup(
  name="lightem",
  version=about["__version__"],
  description="A light-weight python package for multi-table entity matching.",
  packages=find_packages(),  # Ajuste aqui
  package_dir={},  # Ajuste para o diretório 'lightem'
  include_package_data=False,
  long_description=long_description,
  long_description_content_type="text/markdown",
  url=about["__url__"],
  author=about["__author__"],
  author_email="<pedromendicino25@gmail.com>",
  license="MIT",
  classifiers=[
    "Development Status :: 1 - Planning",
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
  ],
  install_requires=[
    "pandas",
    "numpy",
    "scikit-learn",
    "gensim",
    "sentence-transformers",
  ],
  extras_require={
    "dev": ["twine>=6.1.0",],
  },
  python_requires='>=3.10',
)
