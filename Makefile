.ONESHELL:
SHELL := /bin/bash
SRC = $(wildcard nbs/*.ipynb)

all: build docs

diff:
	nbdev_diff_nbs

clean:
	nbdev_clean_nbs

build: $(SRC)
	nbdev_build_lib
	touch build

sync:
	nbdev_update_lib

docs_serve: docs
	cd docs && bundle exec jekyll serve

docs: $(SRC)
	nbdev_build_docs
	touch docs

test:
	nbdev_test_nbs

release: pypi #conda_release
	nbdev_bump_version

conda_release:
	fastrelease_conda_package

pypi: dist
	twine upload --repository pypi dist/*

dist: rm_dist
	python setup.py sdist bdist_wheel

rm_dist:
	rm -rf dist