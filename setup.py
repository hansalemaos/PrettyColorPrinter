from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.1'
DESCRIPTION = "With PrettyColorPrinter, you can print numpy arrays / pandas dataframe / list / dicts / tuple! Shows the path to all items! It even works with nested objects."

# Setting up
setup(
    name="PrettyColorPrinter",
    version=VERSION,
    author="hansalemao",
    author_email="<aulasparticularesdealemaosp@gmail.com>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=['numpy', 'varname', 'regex', 'pandas', 'cprinter', 'input-timeout'],
    keywords=['printer', 'color', 'print', 'pretty print', 'pandas', 'numpy', 'list', 'dict'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)

#python setup.py sdist bdist_wheel
#twine upload dist/*