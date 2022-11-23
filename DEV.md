# Development Guide

## Environment setup

For local development, simply run:
```shell
python setup.py develop
```

Publish to PyPI

```shell
python -m build
# Upload both source and wheel.
twine upload dist/*
```

You could also 
