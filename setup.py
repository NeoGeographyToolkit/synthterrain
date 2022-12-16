#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

requirements = [
    "setuptools"
]

setup(
    entry_points={
        'console_scripts': [
            'synthcraters=synthterrain.crater.cli:main',
            'synthrocks=synthterrain.rock.cli:main',
            'synthterrain=synthterrain.cli:main',
            'craterplot=synthterrain.crater.cli_plot:main',
            'craterconvert=synthterrain.crater.cli_convert:main',
        ],
    },
    install_requires=requirements,
    # include_package_data=True,
    # package_data={
    #     "vipersci": ["data/*"],
    # },
    # packages=find_packages(
    #     include=['synthterrain', 'synthterrain.*'],
    #     where="src/python",
    # ),
    test_suite="tests/python",
    zip_safe=False,
    # package_dir={"": "src/python"},
    tests_require=["pytest"],
)
