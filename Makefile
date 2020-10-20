# Test Makefile for Alexandria project

PYTHON ?= python
PIP ?= pip

all: build install

build:
	$(PYTHON) setup.py sdist

install:
	$(PIP) install -e .