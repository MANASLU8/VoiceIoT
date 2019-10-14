#!/bin/bash

python -m scripts.snips-nlu.normalize
python -m scripts.snips-nlu.split-slots
python -m scripts.snips-nlu.test-slots