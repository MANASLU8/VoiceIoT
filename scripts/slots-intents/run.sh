#!/bin/bash

python -m scripts.slots-intents.normalize
python -m scripts.slots-intents.split-slots
python -m scripts.slots-intents.test-slots