#!/bin/bash

pytext train < vendor/pytext/bilstm.json
python -m scripts.pytext.test-extended