# VoiceIoT
![plot](https://raw.githubusercontent.com/MANASLU8/VoiceIoT/master/images/w2v/cv.png)
![test-plot](https://raw.githubusercontent.com/MANASLU8/VoiceIoT/master/images/w2v/cv-test.png)
# Scripts
All scripts are executed as parts of a python module, for example:  
`python -m scripts.pytext.normalize [-c config.yaml]`  
Here `config.yaml` is a config file containing configuration for scripts such as paths to datasets and required size of the test subset.  
To install dependencies it is required to run  
`cat scripts/requirements.txt | xargs -n 1 pip install`  
## snips-nlu
### Prerequisites
Besides installing dependepcies it is required to manually install language resources like this:  
`python -m snips_nlu download en`  
## pytext
### Prerequisites
It is recommended to install [nvidia apex](https://github.com/NVIDIA/apex/).  
### Training
Training is performed using command  
`pytext train < vendor/pytext/docnn.json`
## SF-ID-Network-For-NLU
**This project uses tensorflow 1.14.0**
### Prerequsites
Clone project to the vendor folder  
```shell
git clone git@github.com:ZephyrChenzf/SF-ID-Network-For-NLU.git
```
Copy dataset to the cloned folder  
```shell
cp dataset/joint-nlu-2 vendor/SF-ID-Network-For-NLU/data/itmo-2
```
### Training
Run script being in the root of the cloned folder  
```shell
python train.py --dataset=itmo-2 --priority_order=slot_first --embedding_path=../../models/w2v/embeddings.npy --use_crf=True
```
## Vowpal Wabbit
### Usage  
#### Train:  
```shell
vw dataset/vw/train.txt --oaa 5 -f models/vw/oaa.model
```  
#### Test:  
```shell
vw -t -i models/vw/oaa.model dataset/vw/test.txt -p dataset/vw/predictions.txt
```
## Parlai
Commands bellow assume that parlai is located in your home folder.  
There is a separate [repository](https://github.com/zeionara/viot) for the used task.
### Usage
#### Train  
```ssh
python ~/parlai/examples/train_model.py -m seq2seq -t viot -bs 64 -eps 2 -mf models/parlai/model
```
#### Eval  
```ssh
python ~/parlai/examples/eval_model.py -m ir_baseline -t viot -dt valid -mf models/parlai/model
```
#### Display model and prediction results for 100 episodes
```ssh
python ~/parlai/examples/display_model.py -t viot -mf models/parlai/model -n 100
```