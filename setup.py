from setuptools import setup

setup(
      name='imrec',
      version='0.1',
      description='Example python program',
      url='https://github.com/ashwinmr/py_example',
      author='Ashwin Rao',
      author_email='iashwinrao@gmail.com',
      license='MIT',
      packages=['imrec'],
      include_package_data=True,
      entry_points = {
            'console_scripts': ['imrec = imrec.imrec:main']
      },
      zip_safe=False
)
