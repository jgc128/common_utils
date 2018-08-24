import re
import logging
import string

import numpy as np

from common_utils.text.vocab import Vocab


class Field(object):
    def __init__(self, init_token='<sos>', eos_token='<eos>', pad_token='<pad>', unk_token='<unk>',
                 padding=True, max_len=50, tokenizer=None, vocab=None, append_eos=True):
        super(Field, self).__init__()

        self.pad_token = pad_token
        self.unk_token = unk_token
        self.init_token = init_token
        self.eos_token = eos_token

        self.append_eos = append_eos

        self.padding = padding
        self.tokenizer = tokenizer or self._default_tokenizer

        self.max_len = max_len

        self.vocab = vocab

    def _default_tokenizer(self, sentence):
        return sentence.split(' ')

    def tokenize_sentence(self, sentence):
        if not isinstance(sentence, list):
            sentence = self.tokenizer(sentence)

        return sentence

    def preprocess(self, sentence):
        # sentence = sentence.replace('-LRB-', '(').replace('-RRB-', ')').replace('-LSB-', '[').replace('-RSB-', ']')
        # sentence = sentence.replace('``', '"').replace("''", '"')
        #
        # sentence = re.sub(r'[^a-zA-Z!.,? ]+', ' ', sentence)
        # sentence = re.sub(r'<[^<]+?>', '', sentence)

        sentence = self.tokenize_sentence(sentence)

        sentence = [token.lower() for token in sentence]

        return sentence

    def process(self, sentence):
        if self.append_eos:
            sentence = sentence[:self.max_len - 1]
            sentence.append(self.eos_token)
        else:
            sentence = sentence[:self.max_len]

        if self.padding:
            needed_pads = self.max_len - len(sentence)
            if needed_pads > 0:
                sentence = sentence + [self.pad_token] * needed_pads

        sentence = [
            self.vocab[token] if token in self.vocab else self.vocab[self.unk_token]
            for token in sentence
        ]

        sentence = np.array(sentence, dtype=np.long)

        return sentence

    def create_vocab(self):
        self.vocab = Vocab(special_tokens=[self.pad_token, self.unk_token, self.eos_token, self.init_token])

    def build_vocab(self, sentences):
        if self.vocab is None:
            self.create_vocab()

        self.vocab.add_documents(sentences)

        logging.info(f'Vocab: {len(self.vocab)}')

    def get_sentence_from_indices(self, sentence_indices, join=True):
        tokens = []
        for idx in sentence_indices:
            token = self.vocab.id2token[idx]

            if token == self.eos_token:
                break

            if token in self.vocab.special_tokens:
                continue

            tokens.append(token)

        if join:
            tokens = ' '.join(tokens)

        return tokens

    def __str__(self):
        return f'Field: {self.max_len} max len, {self.vocab}'
