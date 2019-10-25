#!/bin/bash

python -m scripts.pytext.normalize
python -m scripts.pytext.split
pytext train < vendor/pytext/docnn.json
python -m scripts.pytext.test