#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

package_name = 'synthterrain'
author = 'Ross Beyer'
author_email = 'ross.a.beyer@nasa.gov'

requirements = [
    "setuptools"
]

setup(
    name=package_name,
    version='0.1.0',
    author=author,
    author_email=author_email,
    maintainer=author,
    maintainer_email=author_email,
    entry_points={
        'console_scripts': [
            'synthcraters=synthterrain.crater.cli:main',
            'synthrocks=synthterrain.rock.cli:main',
            'synthterrain=synthterrain.cli:main',
            'craterplot=synthterrain.crater.cli_plot:main',
            'craterconvert=synthterrain.crater.cli_convert:main',
        ],
    },
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=requirements,
    include_package_data=True,
    # package_data={
    #     "vipersci": ["data/*"],
    # },
    packages=find_packages(
        where="src/python",
    ),
    test_suite="tests/python",
    zip_safe=False,
    package_dir={"": "src/python"},
    tests_require=["pytest"],
)
