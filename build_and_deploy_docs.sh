#!/bin/sh

pushd docs
make html
ghp-import -n -p _build/html/
popd
