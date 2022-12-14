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

``craterconvert``
    Converts between the current crater CSV and old XML MATLAB formats.

``craterplot``
    Generates a set of plots from the CSV output of ``synthcraters``.

Contributing
------------

Feedback, issues, and contributions are always gratefully welcomed. See the
contributing guide for details on how to help and setup a development
environment.


