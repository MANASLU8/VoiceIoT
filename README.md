# VoiceIoT
![plot](https://raw.githubusercontent.com/MANASLU8/VoiceIoT/master/images/w2v/cv.png)
![test-plot](https://raw.githubusercontent.com/MANASLU8/VoiceIoT/master/images/w2v/cv-test.png)
# Slots extraction
## snips_nlu
### Perform data normalization
```ssh
python -m scripts.snips-nlu.normalize
```
### Perform splitting data into test and train subsets
If it is required to compare intents:  
```ssh
python -m scripts.snips-nlu.split
```
For preserving slots therewith:  
```ssh
python -m scripts.snips-nlu.split-slots
```
### Perform training and subsequent testing
If it is required to compare intents (on the stage of splitting appropriate script should be executed):  
```ssh
python -m scripts.snips-nlu.test
```
For calculating recall on recognized slots:  
```ssh
python -m scripts.snips-nlu.test-slots
```
### Results
For intents comparison:  
```
Sample                                                                          	recognized-label    	true-label          	correctly-recognized          
закрой занавески если слишком холодно                                           	LightSensor         	TemperatureSensor   	False
если меня нет дома не включай кондиционер                                       	LightSensor         	TemperatureSensor   	False
сколько градусов на улице                                                       	TemperatureSensor   	TemperatureSensor   	True
какая сегодня погода                                                            	MotionSensor        	TemperatureSensor   	False
если на улице холодно включи отопление                                          	HeaterController    	TemperatureSensor   	False
очень жарко                                                                     	SmartFeeder         	TemperatureSensor   	False
когда будет готова еда в духовке                                                	DoorWindowStatusSensor	SmartOven           	False
сколько часов стоит суп в духовке                                               	SmartKettle         	SmartOven           	False
приготовь мясо к нашему ужину                                                   	SmartBreadMaker     	SmartOven           	False
прочитай передачи тогда я засну                                                 	WaterTapRelay       	AudioSystem         	False
назови самую частую песню                                                       	Phone               	AudioSystem         	False
отключи песню если я лягу                                                       	TVSet               	AudioSystem         	False
отключи музыку если я лягу                                                      	TVSet               	AudioSystem         	False
мой любимый альбом                                                              	WaterSupplySensor   	AudioSystem         	False
включи аудиосистему если я приду домой                                          	AudioSystem         	AudioSystem         	True
отключи песню когда я лягу                                                      	SmartKettle         	AudioSystem         	False
увеличь громкость на 4 пункта                                                   	Humidifier          	AudioSystem         	False
включи аудиосистему когда я приду домой                                         	Lamp                	AudioSystem         	False
корми пса три раза в день                                                       	SmartFeeder         	SmartFeeder         	True
если песик давно не ел покорми его                                              	SmartFeeder         	SmartFeeder         	True
включай отопление в комнате                                                     	Lamp                	HeaterController    	False
закрыть дверь если холодно                                                      	WindowRelay         	HeaterController    	False
включай отопление в спальне                                                     	SmartElectricityMeter	HeaterController    	False
поддерживай влажность в комнате на уровне 60                                    	Humidifier          	HumiditySensor      	False
ежели будет жарко закрой окно                                                   	WindowRelay         	WindowRelay         	True
закрой шторы если слишком светло                                                	LightSensor         	WindowRelay         	False
в комнате душно проветри                                                        	Lamp                	WindowRelay         	False
приготовь хлеб к моему приходу                                                  	SmartOven           	SmartBreadMaker     	False
закрой шторы в 9 вечера                                                         	SmartBreadMaker     	CurtainsRelay       	False
если за окном солнце открой занавески                                           	CurtainsRelay       	CurtainsRelay       	True
откроешь занавески если слишком темно                                           	CurtainsRelay       	CurtainsRelay       	True
выключай свет при моем уходе из комнаты                                         	LightSensor         	LightSensor         	True
выключай свет если я ухожу из спальни                                           	LightSensor         	LightSensor         	True
выключай свет после моего ухода из спальни                                      	LightSensor         	LightSensor         	True
выключай свет если я ухожу из комнаты                                           	LightSensor         	LightSensor         	True
освети комнату                                                                  	SmartElectricityMeter	LightSensor         	False
выключи свет                                                                    	SmartToaster        	LightSensor         	False
выключай свет когда я выйду из спальной                                         	LightSensor         	LightSensor         	True
проветри все комнаты                                                            	DoorWindowStatusSensor	DoorWindowStatusSensor	True
какие из окон в доме теперь открыты                                             	DoorWindowStatusSensor	DoorWindowStatusSensor	True
если становится жарко закрой дверь                                              	WindowRelay         	DoorWindowStatusSensor	False
выключай свет если я выхожу из спальни                                          	LightSensor         	Lamp                	False
включи электричество в спальне                                                  	Lamp                	Lamp                	True
выключай свет каждый раз когда я ухожу из гостиной                              	SmartFridge         	Lamp                	False
включай свет когда я захожу в гостиную                                          	Lamp                	Lamp                	True
когда я вернусь приготовь чай                                                   	SmartKettle         	SmartKettle         	True
когда я зайду сделай чай                                                        	SmartKettle         	SmartKettle         	True
включай стиральную машину в 2305 каждое четное число месяца                     	SmartCar            	WashingMachine      	False
включай стиральную машину по четвергам                                          	TVSet               	WashingMachine      	False
если меня нет дома не мой пол                                                   	AudioSystem         	SmartVacuumCleaner  	False
подмети там пока я буду на кухне                                                	SmartVacuumCleaner  	SmartVacuumCleaner  	True
делай чуть тише                                                                 	-                   	TVSet               	False
пульт                                                                           	WaterSupplySensor   	TVSet               	False
запиши программу если я усну                                                    	TVSet               	TVSet               	True
активность в доме                                                               	SmartElectricityMeter	MotionSensor        	False
предупреди если автомобиль сзади меня придется очень вплотную                   	HeatingFloor        	MotionSensor        	False
сообщи был ли ктото дома сегодня                                                	MotionSensor        	MotionSensor        	True
проверь наличие еды в холодильнике                                              	CoffeeMachine       	SmartFridge         	False
сколько дней стоит салат в холодильнике                                         	SmartMicrowave      	SmartFridge         	False
какие у нас есть орехи                                                          	SmartFridge         	SmartFridge         	True
когда я зайду сделай кофе                                                       	Lamp                	CoffeeMachine       	False
подогрей кофе завтра к полуночи                                                 	SmartBreadMaker     	CoffeeMachine       	False
выключай электричество в комнате                                                	Lamp                	SmartElectricityMeter	False
сколько в прошлом месяце было за электричество                                  	SmartElectricityMeter	SmartElectricityMeter	True
сколько в этом году я заплатил за свет                                          	SmartElectricityMeter	SmartElectricityMeter	True
при появлении кота на газоне включи распыление воды 3 раза за раз               	WashingMachine      	WaterTapRelay       	False
я хочу принять ванну                                                            	TVSet               	WaterTapRelay       	False
если на улице гораздо количество градусов включи водоснабжение                  	HeatingFloor        	WaterTapRelay       	False
сделай напор сильнее                                                            	-                   	WaterTapRelay       	False
включен ли утюг                                                                 	SmartOven           	SmartIron           	False
доля углекислого газа                                                           	-                   	null                	False
способы оплаты                                                                  	-                   	null                	False
мне надо проехать до число больницы                                             	SmartCar            	SmartCar            	True
проедь 500 метров                                                               	-                   	SmartCar            	False
распечатай документ название через число секунд                                 	Printer             	Printer             	True
отправь бланк название через число секунд                                       	Printer             	Printer             	True
Correctly recognized 27 of 76 (35.53 %)

```
For slots comparison:  
```ssh
Average recall is 0.6042
```
## slots-intents
**These scripts require python 3.6 and tensorflow 2.0 because of tensorflow-addons' preferences**
### Perform data normalization
```ssh
python -m scripts.slots-intents.normalize
```
### Perform splitting data into test and train subsets
```ssh
python -m scripts.slots-intents.split
```
### Perform training and subsequent testing
```ssh
python -m scripts.slots-intents.test
```
## pytext
### Perform data normalization
```ssh
python -m scripts.pytext.normalize-extended
```
### Perform splitting data into test and train subsets
```ssh
python -m scripts.pytext.split-extended
```
### Perform testing
```ssh
python -m scripts.pytext.test-extended
```
### Results
```
Sample                                                                          	recognized-label    	true-label          	correctly-recognized          
закрыть дверь если начался снег                                                 	CurtainsRelay       	HeaterController    	False
включай отопление в комнате                                                     	HeaterController    	HeaterController    	True
включи воду                                                                     	WaterSupplySensor   	WaterTapRelay       	False
отключи водопровод в гостинной                                                  	HeaterController    	WaterTapRelay       	False
набери воду в ванну к нашему приходу                                            	SmartBreadMaker     	WaterTapRelay       	False
когда последний раз снимали занавески                                           	CurtainsRelay       	CurtainsRelay       	True
если за окном рассвет открой занавески                                          	CurtainsRelay       	CurtainsRelay       	True
открой жалюзи если слишком темно                                                	CurtainsRelay       	CurtainsRelay       	True
если на улице жарко открой окно                                                 	TemperatureSensor   	WindowRelay         	False
проветрите здесь пока я буду в комнате                                          	SmartVacuumCleaner  	WindowRelay         	False
ежели будет жарко закрой окно                                                   	WindowRelay         	WindowRelay         	True
когда я поеду сделай кофе                                                       	SmartKettle         	CoffeeMachine       	False
проверь состояние кофеварки                                                     	MotionSensor        	CoffeeMachine       	False
выключи свет                                                                    	LightSensor         	Lamp                	False
включай солнце когда я захожу в комнату                                         	Lamp                	Lamp                	True
выключай свет когда я иду из санузла                                            	Lamp                	Lamp                	True
включай электричество в комнате                                                 	Lamp                	SmartElectricityMeter	False
сколько в этом месяце было за электричество                                     	SmartElectricityMeter	SmartElectricityMeter	True
сколько дней стоит суп в духовке                                                	SmartOven           	SmartOven           	True
приготовь мне какиенибудь овощи                                                 	SmartBreadMaker     	SmartOven           	False
приготовь хлеб                                                                  	SmartBreadMaker     	SmartBreadMaker     	True
прочитай передачи тогда я засну                                                 	AudioSystem         	AudioSystem         	True
увеличь громкость чтобы не мешал стук                                           	AudioSystem         	AudioSystem         	True
выключи динамики на кухне и в спальне                                           	AudioSystem         	AudioSystem         	True
отключи музыку если я лягу                                                      	AudioSystem         	AudioSystem         	True
включи стереосистему если я приду домой                                         	AudioSystem         	AudioSystem         	True
как сейчас холодно                                                              	WindowRelay         	TemperatureSensor   	False
поддерживай температуру в комнате на уровне число                               	HumiditySensor      	TemperatureSensor   	False
мне сейчас так жарко                                                            	TemperatureSensor   	TemperatureSensor   	True
если меня нет дома не включай кондиционер                                       	SmartVacuumCleaner  	TemperatureSensor   	False
запиши передачу если я засну                                                    	AudioSystem         	TVSet               	False
включи телевизор                                                                	HeaterController    	TVSet               	False
пропылесось здесь пока я буду на кухне                                          	SmartVacuumCleaner  	SmartVacuumCleaner  	True
подмети здесь пока я буду на кухне                                              	SmartVacuumCleaner  	SmartVacuumCleaner  	True
ктонибудь был дома сегодня днем                                                 	MotionSensor        	MotionSensor        	True
снимай парктроник когда я иду домой                                             	SmartKettle         	MotionSensor        	False
какие у нас есть орехи                                                          	SmartFridge         	SmartFridge         	True
проверь наличие еды в холодильнике                                              	MotionSensor        	SmartFridge         	False
корми кота шесть раз в день                                                     	SmartFeeder         	SmartFeeder         	True
насыпай коту меньше корма                                                       	TemperatureSensor   	SmartFeeder         	False
поставь в расписание стирку на 6 утра                                           	WashingMachine      	WashingMachine      	True
включай стиральную машину по четвергам                                          	WashingMachine      	WashingMachine      	True
какие из окон в квартире сейчас открыты                                         	AudioSystem         	DoorWindowStatusSensor	False
выключай свет если я ухожу из прихожей                                          	Lamp                	LightSensor         	False
выключай свет вдруг я выйду из комнаты                                          	SmartElectricityMeter	LightSensor         	False
закрыть окно если начался дождь                                                 	WindowRelay         	HumiditySensor      	False
Correctly recognized 23 of 46 (50.0 %)
```
### Train model

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