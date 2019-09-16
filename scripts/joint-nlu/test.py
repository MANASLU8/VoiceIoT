from .. import file_operators as fo, utils
from jonze import train, test

config = utils.load_config(utils.parse_args().config)

test(dataset = "joint-nlu", datasets_root = config['paths']['datasets']['root'], models_root = config['paths']['models']['joint-nlu'], layer_size=12, batch_size=46)