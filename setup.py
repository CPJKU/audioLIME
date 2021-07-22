from setuptools import setup, find_packages

setup(name='audioLIME',
      version='0.0.3',
      description='audioLIME: Listenable Explanations Using Source Separation',
      url='https://github.com/CPJKU/audioLIME',
      author='Verena Haunschmid',
      author_email='verena.haunschmid@jku.at',
      license='BSD',
      packages= find_packages(exclude=['js', 'node_modules', 'tests']),
      install_requires=[
          'numpy',
          'scipy',
          'librosa>=0.8',
          'numba>=0.52.0',
          'scikit-learn>=0.18',
          'regressors'
      ],
      include_package_data=True,
      zip_safe=False)
