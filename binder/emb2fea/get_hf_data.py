import datasets as ds


import numpy as np
from pathlib import Path
from tqdm import tqdm
from apache_beam.runners import DirectRunner

from loader import load_data  # , load_embeddings, assign_emb_dataset, generate_random_embs


# English
wiki_en = ds.load_dataset('wikipedia', language='simple', date='20230620', beam_runner=DirectRunner())
# wiki_en = ds.load_dataset('wikipedia', language='en', date='20230620', beam_runner=DirectRunner())

binder_en = load_data(Path('../../data/binder_dataset.txt'))

exit(0)

# Chinese
wiki_zh = ds.load_dataset('wikipedia', language='zh', date='20230620', beam_runner=DirectRunner())




