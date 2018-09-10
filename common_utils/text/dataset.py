from common_utils.text import Field, Vocab


class Dataset(object):
    PAD_TOKEN = '<pad>'
    UNK_TOKEN = '<unk>'
    INIT_TOKEN = '<sos>'
    EOS_TOKEN = '<eos>'

    def __init__(self, sentences, fields=None, max_lengths=None, vocab_max_size=0, tokenizer=None):
        super().__init__()

        if max_lengths is None:
            max_lengths = 0
        if not isinstance(max_lengths, (tuple, list)):
            max_lengths = [max_lengths, ] * len(sentences)

        if fields is None:
            self.vocab = Vocab(
                special_tokens=[Dataset.PAD_TOKEN, Dataset.INIT_TOKEN, Dataset.EOS_TOKEN, Dataset.UNK_TOKEN]
            )
            self.fields = [
                Field(
                    init_token=Dataset.INIT_TOKEN, eos_token=Dataset.EOS_TOKEN, pad_token=Dataset.PAD_TOKEN,
                    unk_token=Dataset.UNK_TOKEN, padding=True, max_len=max_len, tokenizer=tokenizer, vocab=self.vocab
                )
                for max_len in max_lengths
            ]
        else:
            self.fields = fields
            self.vocab = self.fields[0].vocab

        self.sentences = [[field.preprocess(s) for s in sents] for field, sents in zip(self.fields, sentences)]

        if fields is None:
            for sents in self.sentences:
                self.vocab.add_documents(sents)

            if vocab_max_size != 0:
                self.vocab.prune_vocab(min_count=2, max_size=vocab_max_size)

    def __len__(self):
        return len(self.sentences[0])

    def __getitem__(self, index):
        sents = [self.sentences[k][index] for k in range(len(self.fields))]
        sents = [field.process(sent) for field, sent in zip(self.fields, sents)]

        return sents
