.. highlight:: shell

============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

For a high-level overview of the philosophy of contributions, please see
https://github.com/planetarypy/TC/blob/master/Contributing.md.

You can contribute in many ways:

Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/https://github.com/NeoGeographyToolkit/synthterrain/issues .

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

Write Documentation
~~~~~~~~~~~~~~~~~~~

This software could always use more documentation, whether as part of the
official docs, in docstrings, or even on the web in blog posts,
articles, and such.

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at https://github.com/NeoGeographyToolkit/synthterrain/issues

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

Get Started!
------------

Ready to contribute? Here's how to set up `synthterrain` for local development.

1. Fork the `synthterrain` repo on GitHub.
2. Clone your fork locally::

    $> git clone git@github.com:your_name_here/synthterrain.git

3. Install your local copy into a virtual environment of your choice (there are many to choose from like conda, etc.). We will assume conda here, but any should work::

    $> cd synthterrain/
    $> conda env create -n synthterrain
    $> conda activate synthterrain
    $> mamba env update --file environment_dev.yml
    $> mamba env update --file environment.yml
    $> pip install --no-deps -e .

   The last ``pip install`` installs synthterrain in "editable" mode which facilitates using the programs and testing.

4. Create a branch for local development::

    $> git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

5. When you're done making changes, check that your changes pass flake8 and the
   tests.

    $> make lint
    $> make test


6. Commit your changes and push your branch to GitHub::

    $> git add .
    $> git commit -m "Your detailed description of your changes."
    $> git push origin name-of-your-bugfix-or-feature

7. Submit a pull request through the GitHub website.

Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the list in CHANGELOG.rst.
3. The pull request should work for Python 3.8, 3.9, 3.10, 3.11 and optionally for PyPy.
   And make sure that the tests pass for all supported Python versions.

What to expect
--------------

Our development of synthterrain is not particularly continuous,
and it is entirely possible that when you submit a PR
(pull request), none of us will have the time to evaluate or integrate
your PR.  If we don't, we'll try and communicate that with you via the
PR.

For large contributions, it is likely that you, or your employer,
will be retaining your copyrights, but releasing the contributions
via an open-source license.  It must be compatible with the Apache-2
license that synthterrain is distributed with, so that we can redistribute
that contribution with synthterrain, give you credit, and make synthterrain even
better!  Please contact us if you have a contribution of that nature,
so we can be sure to get all of the details right.

For smaller contributions, where you (or your employer) are not
concerned about retaining copyright (but we will give you credit!),
you will need to fill out a Contributor License Agreement (CLA)
before we can accept your PR.  The CLA assigns your copyright in
your contribution to NASA, so that our NASA copyright statement
remains true:

    Copyright (c) YEAR, United States Government as represented by the
    Administrator of the National Aeronautics and Space Administration.
    All rights reserved.

There is an `Individual CLA
<https://github.com/NeoGeographyToolkit/synthterrain/blob/master/docs/synthterrain_ARC-18971-1_Individual_CLA.pdf>`_ and a
`Corporate CLA
<https://github.com/NeoGeographyToolkit/synthterrain/blob/master/docs/synthterrain_ARC-18971-1_Corporate_CLA.pdf>`_.

synthterrain People
-------------------

- A synthterrain **Contributor** is any individual creating or commenting
  on an issue or pull request.  Anyone who has authored a PR that was
  merged should be listed in the AUTHORS.rst file.

- A synthterrain **Committer** is a subset of contributors, typically NASA
  employees or contractors, who have been given write access to the
  repository.

Rules for Merging Pull Requests
-------------------------------

Any change to resources in this repository must be through pull
requests (PRs). This applies to all changes to documentation, code,
binary files, etc. Even long term committers must use pull requests.

In general, the submitter of a PR is responsible for making changes
to the PR. Any changes to the PR can be suggested by others in the
PR thread (or via PRs to the PR), but changes to the primary PR
should be made by the PR author (unless they indicate otherwise in
their comments). In order to merge a PR, it must satisfy these conditions:

1. Have been open for 24 hours.
2. Have one approval.
3. If the PR has been open for 2 days without approval or comment, then it
   may be merged without any approvals.

Pull requests should sit for at least 24 hours to ensure that
contributors in other timezones have time to review. Consideration
should also be given to weekends and other holiday periods to ensure
active committers all have reasonable time to become involved in
the discussion and review process if they wish.

In order to encourage involvement and review, we encourage at least
one explicit approval from committers that are not the PR author.

However, in order to keep development moving along with our low number of
active contributors, if a PR has been open for 2 days without comment, then
it could be committed without an approval.

The default for each contribution is that it is accepted once no
committer has an objection, and the above requirements are
satisfied.

In the case of an objection being raised in a pull request by another
committer, all involved committers should seek to arrive at a
consensus by way of addressing concerns being expressed by discussion,
compromise on the proposed change, or withdrawal of the proposed
change.

Exceptions to the above are minor typo fixes or cosmetic changes
that don't alter the meaning of a document. Those edits can be made
via a PR and the requirement for being open 24 h is waived in this
case.


.. Deploying
   ---------
   
   A reminder for the maintainers on how to deploy.
   Make sure all your changes are committed (including an entry in CHANGELOG.rst).
   Then run::
   
   $ bump2version patch # possible: major / minor / patch
   $ git push
   $ git push --tags
