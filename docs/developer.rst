
Development
===========

Documentation
-------------

To rebuild the documentation, install several packages::

    pip install -r docs/requirements.txt

And then run from the ``docs`` directory::

    make html

This can take a long time because it executes many of the examples;
if you'd prefer to build the docs without executing the notebooks, you can run::

    make html-noplot

You can then see the generated documentation in ``docs/_build/html/index.html``.



Update notebooks
++++++++++++++++

We use `jupytext <https://jupytext.readthedocs.io>`_ to maintain two synced copies of the notebooks
in ``docs/notebooks``: one in ``ipynb`` format, and one in ``md`` format. The advantage of the former
is that it can be opened and executed directly in Colab; the advantage of the latter is that
it makes it much easier to track diffs within version control.

Editing ipynb
+++++++++++++

For making large changes that substantially modify code and outputs, it is easiest to
edit the notebooks in Jupyter or in Colab. To edit notebooks in the Colab interface,
open http://colab.research.google.com and ``Upload`` from your local repo.
Update it as needed, ``Run all cells`` then ``Download ipynb``.
You may want to test that it executes properly, using ``sphinx-build`` as explained above.

Editing md
++++++++++

For making smaller changes to the text content of the notebooks, it is easiest to edit the
``.md`` versions using a text editor.

Syncing notebooks
+++++++++++++++++

After editing either the ipynb or md versions of the notebooks, you can sync the two versions
using `jupytext <https://jupytext.readthedocs.io>`_. For example, to sync the files inside the ``docs/notebooks/deep_learning/``, run the command::

    jupytext --sync docs/notebooks/deep_learning/*.*


Be sure to use the version of jupytext specified in
`.pre-commit-config.yaml <https://github.com/google/jaxopt/blob/main/.pre-commit-config.yaml>`_.

Alternatively, you can use the `pre-commit <https://pre-commit.com>`_ framework to run this
on all staged files in your git repository, automatically using the correct jupytext version::

    pre-commit run jupytext

See the pre-commit framework documentation for information on how to set your local git
environment to execute this automatically.

Creating new notebooks
++++++++++++++++++++++

If you are adding a new notebook to the documentation and would like to use the ``jupytext --sync``
command discussed here, you can set up your notebook for jupytext by using the following command::

    jupytext --set-formats ipynb,md:myst path/to/the/notebook.ipynb


This works by adding a ``"jupytext"`` metadata field to the notebook file which specifies the
desired formats, and which the ``jupytext --sync`` command recognizes when invoked.

Notebooks within the sphinx build
+++++++++++++++++++++++++++++++++

We exclude some notebooks from the build, e.g., because they contain long computations.
See ``exclude_patterns`` in `conf.py <https://github.com/google/jaxopt/blob/main/docs/conf.py>`_.
