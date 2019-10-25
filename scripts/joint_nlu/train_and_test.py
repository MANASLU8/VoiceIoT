from .. import file_operators as fo, utils
from jonze import train, test

config = utils.load_config(utils.parse_args().config)

for layer_size in range(1, 64):
	print(f"layer_size: {layer_size}")
	train(dataset = "joint-nlu", datasets_root = config['paths']['datasets']['root'], models_root = config['paths']['models']['root'], layer_size=layer_size, batch_size=46, number_of_epochs=5)
	test(dataset = "joint-nlu", datasets_root = config['paths']['datasets']['root'], models_root = config['paths']['models']['joint-nlu'], layer_size=layer_size, batch_size=46)