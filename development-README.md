# Steps to set up testing environment
Set up conda environment:

```
conda create --name dragon_baseline python=3.10
```

Activate environment:

```
conda activate dragon_baseline
```

Install module and additional development dependencies:

```
pip install -e .
pip install -r requirements_dev.txt
```

Perform tests:

```
./build.sh
./test.sh
```

# Programming conventions
AutoPEP8 for formatting (this can be done automatically on save, see e.g. https://code.visualstudio.com/docs/python/editing)

# Push release to PyPI
1. Increase version in setup.py, and set below
2. Build: `python -m build`
3. Distribute package to PyPI: `python -m twine upload dist/*0.4.6*`
