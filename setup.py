from setuptools import find_packages
from setuptools import setup

with open("requirements_dev.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='polclassifier',
      version="0.0.1",
      description="UK Political Speech Classifier",
      license="MIT",
      author="szaboildi",
      author_email="ies236@nyu.edu",
      #url="https://github.com/szaboildi/uk-pol-speech-classifier",
      install_requires=requirements,
      packages=find_packages(),
      test_suite="tests",
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      zip_safe=False)
