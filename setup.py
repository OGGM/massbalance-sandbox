
"""

adapted from https://github.com/fmaussion/scispack;
https://packaging.python.org/en/latest/distributing.html;
https://github.com/pypa/sampleproject

"""


# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


# %%
setup(
    name='MBsandbox',
    version='0.0.1',  # Required
    description='Better mass balance models for OGGM',  # Required
    long_description=long_description,  # Optional

    # Denotes that our long_description is in Markdown; valid values are
    # text/plain, text/x-rst, and text/markdown
    #
    # Optional if long_description is written in reStructuredText (rst) but
    # required for plain-text or Markdown; if unspecified, "applications should
    # attempt to render [the long_description] as text/x-rst; charset=UTF-8 and
    # fall back to text/plain if it is not valid rst" (see link below)
    #
    # This field corresponds to the "Description-Content-Type" metadata field:
    # https://packaging.python.org/specifications/core-metadata/#description-content-type-optional
    long_description_content_type='text/markdown',  # Optional (see note above)
    url='https://github.com/OGGM/massbalance-sandbox',

    # This should be your name or the name of the organization which owns the
    # project.
    author='Lilian Schuster, Fabien Maussion & OGGM contributors',  # Optional

    # This should be a valid email address corresponding to the author listed
    # above.
    author_email='lilian.schuster@student.uibk.ac.at',  # Optional

    # Classifiers help users find your project by categorizing it.
    #
    # For a list of valid classifiers, see https://pypi.org/classifiers/
    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Software Development :: Build Tools',

        # Pick your license as you wish
        'License :: OSI Approved :: BSD License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.7.8',
    ],

    # This field adds keywords for your project which will appear on the
    # project page. What does your project relate to?
    # Note that this is a string of words separated by whitespace, not a list.
    keywords='glaciers climate geosciences',
    # You can just specify package directories manually here if your project is
    # simple. Or you can use find_packages().
    #
    # Alternatively, if you just want to distribute a single Python file, use
    # the `py_modules` argument instead as follows, which will expect a file
    # called `my_module.py` to exist:
    #
    #   py_modules=["my_module"],
    #
    packages=find_packages(exclude=['docs', 'MBsandbox.tests']),
    # , 'massbalance-sandbox.tests']),  # Required

    # This field lists other packages that your project depends on to run.
    # Any package you put here will be installed by pip when your project is
    # installed, so they must be valid existing projects.
    #
    # For an analysis of "install_requires" vs pip's requirements files see:
    # https://packaging.python.org/en/latest/requirements.html

    # the newest oggm developer version has to be installed in order that
    # this works
    install_requires=[],
    # 'numpy', 'pytest', 'matplotlib', 'scipy', 'pandas',
    # 'xarray', 'netCDF4', 'shapely', 'tables', 'geopandas', 'salem',
    # 'joblib', 'descartes','rasterio', 'motionless', 'scikit-image' 'oggm'

    # List additional groups of dependencies here (e.g. development
    # dependencies). Users will be able to install these using the "extras"
    # syntax, for example:
    #
    #   $ pip install sampleproject[dev]
    #
    # Similar to `install_requires` above, these must be valid existing
    # projects.
    extras_require={  # Optional
        # 'test': ['pytest'],
        # 'docs': ['ast']
    },

    python_requires='>=3.7',
    # If there are data files included in your packages that need to be
    # installed, specify them here.
    #package_data={  # Optional
    #},
    #include_package_data=True,
    #package_data={'': ['data/*.csv'], '':['MBsandbox/data/*.csv']},

    # Although 'package_data' is the preferred approach, in some case you may
    # need to place data files outside of your packages. See:
    # http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files
    #
    # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
    data_files={},  # Optional

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # `pip` to create the appropriate form of executable for the target
    # platform.
    #
    # For example, the following would provide a command called `sample` which
    # executes the function `main` from this package when invoked:
    # entry_points={  # Optional
    #    'console_scripts': [
    #        'scispack=scispack.cli:main',
    #    ],
    # },
    # entry_points={  # Optional
    # },
    # List additional URLs that are relevant to your project as a dict.
    #
    # This field corresponds to the "Project-URL" metadata fields:
    # https://packaging.python.org/specifications/core-metadata/#project-url-multiple-use
    #
    # Examples listed include a pattern for specifying where the package tracks
    # issues, where the source is hosted, where to say thanks to the package
    # maintainers, and where to support the project financially. The key is
    # what's used to render the link text on PyPI.
    project_urls={  # Optional
        'Bug Reports': 'https://github.com/OGGM/massbalance-sandbox/issues',
    },
)
