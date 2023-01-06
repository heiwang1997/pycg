# Development Guide

## Environment setup

- For projects / enviroments that would modify / develop pycg, use `develop`:
```shell
python setup.py develop

# Fix linting/IntelliSense [one-time operation, optional]
bash dev_link.sh
```

- For other projects / environments, just install it and occasionally upgrade it for newest features.

## Version bump

1. Modify the version number in `pycg/__init__.py`
2. Run `git commit`.
3. Run `git tag vX.Y.Z`.
4. Run `git push origin vX.Y.Z` to publish the tag.

## Publish to PyPI

- If you are working locally:
```shell
python -m pip install --upgrade build twine
python -m build
# Upload both source and wheel.
twine upload dist/*
```

- Using Github Actions. It is automatically triggered on package release (using web portal of Github).

- If CI/CD failed, either 3 of the following works:
    1. Fix the bug, bump the version, and re-create a release using the new version.
    2. Fix the bug and re-run the Github Action by hand.
    3. Build locally and upload to PyPI.

# Documentation Guide

We use [mkdocs](https://squidfunk.github.io/mkdocs-material) to manage the documentations.
Use `mkdocs serve` or `mkdocs build` under the root directory will work.
Additionally, you may use ReadTheDocs to host it by visiting the [Dashboard](https://readthedocs.org/dashboard/).
