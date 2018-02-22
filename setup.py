from setuptools import setup


def readme():
    with open('README.rst') as f:
        return f.read()

setup(
    name='deconvolution',
    version='1.0.0',
    description='Package performing colour deconvolution',
    url='https://github.com/grfrederic/deconvolution',
    author='Frederic Grabowski, Pawel Czyz',
    author_email='grabowski.frederic@gmail.com, pczyz@protonmail.com',
    license='BSD 3-Clause License',
    packages=['deconvolution'],

    install_requires=['numpy', 'Pillow'],

    # if anybody wants to add non-py files, he must uncomment the following line and add them in MANIFEST.in
    # include_package_data=True
    zip_safe=False
)
