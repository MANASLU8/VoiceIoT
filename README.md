# VoiceIoT
# Scripts
All scripts (including vendor's scripts) are executed from the according folder inside `scripts` directory.  
To install dependencies it is required to run  
`pip install -r requirements.txt`  
being in the `scripts` folder.
## slots-intents
### Prerequisities
For scripts to work it is required that repository [slots_intents](https://github.com/mohammedterry/slots_intents) is clonned to the `vendor` folder:  
`git clone git@github.com:mohammedterry/slots_intents.git`
## pytext
These scripts are executed as parts of a python module, for example:  
`python -m scripts.pytext.normalize [-c config.yaml]`  
Here `config.yaml` is a config file containing configuration for scripts such as paths to datasets and required size of the test subset.
### Prerequisities
It is recommended to install [nvidia apex](https://github.com/NVIDIA/apex/).  