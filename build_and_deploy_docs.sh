#!/bin/sh

# Requires the following:
# -----------------------
# pip install ghp-import
# conda install sphinx
# pip install guzzle_sphinx_theme
# pip install numpydoc
# pip install nbsphinx
# pip install sphinxcontrib-inlinesyntaxhighlight
# conda install ipython
# conda install -c conda-forge pandoc=1.19.2

pushd docs
make html
ghp-import -n -p _build/html/
popd
