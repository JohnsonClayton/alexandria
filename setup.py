from distutils.core import setup

setup(name='alexandria',
      version=0.1,
      description='high-level machine learning library',
      author='Clayton Johnson',
      author_email='claypjay@gmail.com',
      url='https://github.com/JohnsonClayton/alexandria',
      package_dir={'' : 'alexandria'},
      py_modules=['examples', 'experiment', 'models', 'metrics', 'tests']
      )

