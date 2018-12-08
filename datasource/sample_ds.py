import os
import json
import pickle
import random
from typing import List
import numpy as np


class SampleDataSource:

    def __init__(self, config):
        self._config = config
        self._ask, self._res = self.load_data()
        self._create_tokenizer()

    def load_data(self):
        data = json.load(open(os.path.join(self._config.data_path, 'sample_data.json'), 'r'))
        ask = [s['a'] for s in data['dialogue']]
        res = [s['r'] for s in data['dialogue']]
        return ask, res

    def feed_dict(self, model, batch_size: int, is_transformer: bool=False):
        inputs = []
        targets = []

        num_batch = len(self._ask) // batch_size
        batch = []

        for i in range(num_batch):
            start_index = batch_size * i
            end_index = batch_size * (i + 1)
            inputs = self._ask[start_index:end_index]
            targets = self._res[start_index:end_index]
            if is_transformer:
                d = self._create_dict_transformer(model, inputs, targets)
            else:
                d = self._create_dict(model, inputs, targets)
            batch.append(d)

        return batch

    def shuffle(self):
        data = list(zip(self._ask, self._res))
        data = np.array(random.sample(data, len(data)))
        self._ask, self._res = data[:, 0], data[:, 1]

    def _create_dict_transformer(self, model, inputs: List[str], targets: List[str]):
        inputs = self.batch(inputs, suffix=[self.eos_id])
        decoder_inputs = self.batch(targets, prefix=[self.bos_id])
        decoder_targets = self.batch(targets, suffix=[self.eos_id])
        # d = {
        #     'encoder_inputs': inputs,
        #     'decoder_inputs': decoder_inputs,
        #     'decoder_targets': decoder_targets,
        # }
        return np.array([inputs, decoder_inputs, decoder_targets])

    def _create_dict(self, model, inputs: List[str], targets: List[str]):
        inputs = self.batch(inputs, suffix=[self.eos_id])
        inputs_length = [len(input) for input in inputs]
        encoder_targets = self.batch(targets, prefix=[self.bos_id])
        decoder_targets = self.batch(targets, suffix=[self.eos_id])
        encoder_targets_length = [len(target) for target in encoder_targets]
        decoder_targets_length = [len(target) for target in decoder_targets]
        d = {
            model.inputs: inputs,
            model.inputs_length: inputs_length,
            model.encoder_targets: encoder_targets,
            model.encoder_targets_length: encoder_targets_length,
            model.decoder_targets: decoder_targets,
            model.decoder_targets_length: decoder_targets_length
        }
        return d

    def _create_tokenizer(self):
        with open('./data/vocab.pkl', 'rb') as f:
            self._word_to_id, self._id_to_word = pickle.load(f)

    def id_list_to_sentence(self, id_list: List[int]):
        return [self._id_to_word[idx] for idx in id_list]

    def sentence_to_id_list(self, sentence: str):
        return [self._word_to_id[word] for word in sentence]

    def batch(self, batch: List[str], prefix=None, suffix=None):
        prefix = prefix or []
        suffix = suffix or []
        max_length = self._config.max_length if (prefix is None and suffix is None) else self._config.max_length - 1
        batch_list = [prefix + self.sentence_to_id_list(b)[-max_length:] + suffix for b in batch]
        batch_list = [batch + [0] * (self._config.max_length - len(batch)) for batch in batch_list]
        return batch_list

    @property
    def eos_id(self):
        return self._word_to_id['<eos>']

    @property
    def bos_id(self):
        return self._word_to_id['<bos>']

    @property
    def vocab_size(self):
        return len(self._word_to_id)
