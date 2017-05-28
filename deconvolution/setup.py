from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(
    name='deconvolution',
    version='0.0.2',
    description='Package performing colour deconvolution',
    url='todo',
    author='todo',
    author_email='todo@todo.todo',
    license='todo',
    packages=['deconvolution'],

    # Due to http://stackoverflow.com/questions/8710918/installing-numpy-as-a-dependency-with-setuptools the next line is required...
    #setup_requires=['numpy'],
    
    #install_requires=['numpy', 'PIL'],

    # if anybody wants to add non-py files, he must uncomment the following line and add them in MANIFEST.in
    #include_package_data=True
    zip_safe=False
)
