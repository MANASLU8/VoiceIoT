# VoiceIoT
# Scripts
All scripts are executed as parts of a python module, for example:  
`python -m scripts.pytext.normalize [-c config.yaml]`  
Here `config.yaml` is a config file containing configuration for scripts such as paths to datasets and required size of the test subset.  
To install dependencies it is required to run  
`pip install -r requirements.txt`  
being in the `scripts` folder.
## snips-nlu
### Prerequisities
Besides installing dependepcies it is required to manually install language resources like this:  
`python -m snips_nlu download en`  
## slots-intents
### Prerequisities
For scripts to work it is required that repository [slots_intents](https://github.com/mohammedterry/slots_intents) is clonned to the `vendor` folder:  
`git clone git@github.com:mohammedterry/slots_intents.git`
## pytext
### Prerequisities
It is recommended to install [nvidia apex](https://github.com/NVIDIA/apex/).  
