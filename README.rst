============
synthterrain
============

The synthterrain package is software to support the creation of synthetic
terrain on worlds in the solar system.

At the moment, this repo is under significant development and change as we
attempt to craft various pieces of code.  It is very messy and a work-in-process.
Nothing is guaranteed about structure until we pass the 1.0 version.

It currently contains only Python code, but we anticipate the addition of C++
for certain functionality.


Features
--------

The synthterrain package currently offers these command-line programs
when it is installed (see the CONTRIBUTING document).  Arguments
can be found by running any program with a ``-h`` flag.

``synthcraters``
    This program generates synthetic crater populations.

``synthrocks``
    This program generates synthetic rock populations (generally run after
    ``synthcraters`` so that rocks can be placed relative to crater ejecta
    fields.

``synthterrain``
    This program mostly just runs ``synthcraters`` and then immediately runs
    ``synthrocks``.  Also allows a set of pre-existing craters to be added
    to the probabiliy maps that ``synthrocks`` uses to place rocks.

``synthcraterconvert``
    Converts between the current crater CSV and old XML MATLAB formats.

``synthcraterplot``
    Generates a set of plots from the CSV output of ``synthcraters``.


The command lines for these programs can get long, and if you would prefer to
write a text file pre-loaded with the arguments, you can do so with ampersand-arguments.

For example, you could write a text file ``test1.txt`` that looks like this::

    # This is an arguments file, lines that start with octothorpes are ignored.
    -v
    --craters craters_5m_to_500m.csv
    --cr_maxd 5
    --cr_mind 0.5
    --rk_maxd 2
    --rk_mind 0.1
    --probability_map_gsd
    --cr_outfile test_cr.csv
    --rk_outfile test_rk.csv

And then you could call ``synthterrain`` like this::

    $> synthterrain @test1.txt

You can mix regular arguments and ampersand-arguments if you wish.


Installation
------------
TBD.


Contributing
------------

Feedback, issues, and contributions are always gratefully welcomed. See the
contributing guide for details on how to help and setup a development
environment.


Credits
-------

synthterrain was developed in the open at NASA's Ames Research Center.

See the `AUTHORS <https://github.com/NeoGeographyToolkit/syntheterrain/blob/master/AUTHORS.rst>`
file for a complete list of developers.


License
-------
Copyright © 2024, United States Government, as represented by the
Administrator of the National Aeronautics and Space Administration.
All rights reserved.

The "synthterrain" software is licensed under the Apache License,
Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License
at http://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing
permissions and limitations under the License.



