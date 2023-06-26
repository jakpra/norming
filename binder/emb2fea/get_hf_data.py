import datasets as ds


import numpy as np
from pathlib import Path
from tqdm import tqdm
from apache_beam.runners import DirectRunner, DataflowRunner
import random
import pandas
from collections import defaultdict, Counter
import spacy

from loader import load_data  # , load_embeddings, assign_emb_dataset, generate_random_embs


# English
wiki_en = iter(ds.load_dataset('wikipedia', '20220301.en', beam_runner=DataflowRunner(), streaming=True)['train'].shuffle(seed=42, buffer_size=10_000))
outfile = 'wiki-20220301.en_sample.txt'

nlp = spacy.load("en_core_web_sm")

binder_en = pandas.read_csv(Path('../../data/binder_dataset.txt'), delimiter='\t')
target_words = list(enumerate(binder_en['Word'].to_list()))

target_samples = defaultdict(list)
target_sample_counts = Counter({tgt: 0 for tgt in target_words})

ex = next(wiki_en)
sents = iter(nlp(ex['text']).sents)
sent = next(sents).text.strip()
sent_offs = 0
while '\t' in sent or '\n' in sent:
    sent = next(sents).text.strip()
    sent_offs += 1

last_found = None
last_10_buffer = []

random.seed(42)

with tqdm(total=1000*len(target_words)) as pbar:
    while target_sample_counts:
        random.shuffle(target_words)
        for i, tgt in target_words:
            pbar.set_postfix({'total': target_sample_counts.most_common(1),
                              'last found': last_found,
                              'last checked sent': {'id': ex['id'],
                                                    'sent_offs': sent_offs,
                                                    'title': ex['title']},
                              'last checked tgt': f'{i:03d}'})
            tgt = tgt.strip()
            if f' {tgt} ' in sent:
                tgt_smpl = dict(**ex)
                tgt_smpl['text'] = sent
                tgt_smpl['doc_sent_offs'] = str(sent_offs)
                tgt_smpl['tgt'] = tgt
                tgt_smpl['tgt_sent_offs'] = str(target_sample_counts[tgt])
                target_samples[tgt].append(tgt_smpl)
                last_10_buffer.append(tgt_smpl)

                pbar.update(1)
                if pbar.n % 10 == 0:
                    with open(outfile, 'a', encoding='utf-8') as f:
                        for found in last_10_buffer:
                            f.write('\t'.join([found['tgt'],
                                               found['tgt_sent_offs'],
                                               found['id'],
                                               found['title'],
                                               found['doc_sent_offs'],
                                               found['text']]))
                            f.write('\n')
                    last_10_buffer = []

                target_sample_counts[tgt] += 1

                if target_sample_counts[tgt] == 1000:
                    del target_sample_counts[tgt]

                if not target_sample_counts:
                    break

                last_found = {'id': ex['id'],
                              'sent_offs': sent_offs,
                              'title': ex['title'],
                              'tgt': tgt}

            try:
                sent = next(sents).text.strip()
                sent_offs += 1
                while '\t' in sent or '\n' in sent:
                    sent = next(sents).text.strip()
                    sent_offs += 1
            except StopIteration:
                ex = next(wiki_en)
                sents = iter(nlp(ex['text']).sents)
                sent = next(sents).text.strip()
                sent_offs = 0

exit(0)

# Chinese
wiki_zh = ds.load_dataset('wikipedia', language='zh', date='20230620', beam_runner=DirectRunner())




