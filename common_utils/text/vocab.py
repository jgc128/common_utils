import logging
from collections import Counter


class Vocab(object):
    def __init__(self, special_tokens=None):
        super(Vocab, self).__init__()

        self._reset_vocab()

        if special_tokens is None:
            special_tokens = []

        self.special_tokens = special_tokens
        self.add_document(self.special_tokens)

    def _reset_vocab(self, reset_counts=True):
        self.nb_tokens = 0
        self.token2id = {}
        self.id2token = {}

        if reset_counts:
            self.token_counts = Counter()

    def _add_token(self, token):
        if token not in self.token2id:
            self.token2id[token] = self.nb_tokens
            self.id2token[self.nb_tokens] = token
            self.nb_tokens += 1

    def add_document(self, document):
        for token in document:
            self.token_counts[token] += 1
            self._add_token(token)

    def add_documents(self, documents):
        for doc in documents:
            self.add_document(doc)

    def prune_vocab(self, min_count=2, max_size=50000):
        nb_tokens_before = len(self.token2id)

        most_common_tokens = set(t for t, c in self.token_counts.most_common(max_size))
        min_count_tokens = set([t for t, c in self.token_counts.items() if c >= min_count])
        special_tokens = set(self.special_tokens)
        tokens_to_leave = (most_common_tokens & min_count_tokens) | special_tokens

        all_tokens = set(self.token_counts.keys())
        tokens_to_delete = all_tokens - tokens_to_leave

        for token in tokens_to_delete:
            self.token_counts.pop(token)

        self._reset_vocab(reset_counts=False)
        for token in self.special_tokens:
            self._add_token(token)
        for token in self.token_counts.keys():
            self._add_token(token)

        logging.info(f'Vocab pruned: {nb_tokens_before} -> {self.nb_tokens}')

    def __getitem__(self, item):
        return self.token2id[item]

    def __contains__(self, item):
        return item in self.token2id

    def __len__(self):
        return self.nb_tokens

    def __str__(self):
        return f'Vocab: {self.nb_tokens} tokens'
