# encoding: utf-8
"""
@author: wuwei 
@contact: wu.wei@pku.edu.cn

@version: 1.0
@license: Apache Licence
@file: setup.py
@time: 19-1-19 上午10:34


"""
from setuptools import setup, find_packages

setup(name='glyce',
      version='1.0',
      description='Shannon Glyce embedding for Chinese word and character',
      url='https://github.com/ShannonAI/glyce',
      long_description=open("README.md").read(),
      long_description_content_type="text/markdown",
      author='Shannon.AI',
      author_email='Shannon.AI',
      license='Apache License 2.0',
      packages=find_packages(),
      install_requires=[
          'torch',
          'torchvision',
          'Pillow',
          'zhconv==1.4.0',
          'pypinyin==0.34.1',
          'pywubi==0.0.2',
      ],
      python_requires='>=3.6.1',
      zip_safe=False)
