#!/bin/bash

python -m scripts.pytext.normalize-extended
python -m scripts.pytext.split-extended
pytext train < vendor/pytext/bilstm.json
python -m scripts.pytext.test-extended