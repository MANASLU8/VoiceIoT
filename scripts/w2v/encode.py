from .. import utils
from . import normalize
from .. import joint_nlu as jn
import os
import numpy as np
from gensim.models import KeyedVectors
args = utils.parse_args()
config = utils.load_config(args.config)

DEFAULT_MODEL_PATH =  "~/models/ArModel100w2v.txt"#"/home/zeio/viot/models/w2v/test.txt"
DEFAULT_ARRAY_FILE_NAME = "embeddings"

df = normalize.handle_files(jn.normalize.handle_files(utils.list_dataset_files(config['paths']['datasets']['annotations']), get_file_names = True))
model = KeyedVectors.load_word2vec_format(DEFAULT_MODEL_PATH, binary=False)

array_to_save = np.array([i for i in list(map(lambda text: np.average([[value for value in model[word]] for word in text.split(" ") if word in model], axis=0), df.text)) if type(i) != np.float64])

print(f"Resulting shape: {array_to_save.shape}")
np.save(os.path.join(config['paths']['models']['w2v'], DEFAULT_ARRAY_FILE_NAME), array_to_save)