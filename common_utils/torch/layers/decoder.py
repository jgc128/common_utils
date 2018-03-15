import numpy as np
import torch

from common_utils.torch.helpers import variable, argmax


class Decoder(torch.nn.Module):
    def __init__(self, embedding, hidden_size, max_len, init_token, dropout=0.2):
        super(Decoder, self).__init__()

        self.embedding = embedding
        self.hidden_size = hidden_size
        self.dropout_prob = dropout

        self.max_len = max_len
        self.init_token = init_token

        embedding_dim = embedding.weight.size(1)
        vocab_size = embedding.weight.size(0)
        self.decoder = torch.nn.GRUCell(input_size=embedding_dim, hidden_size=hidden_size)
        self.out = torch.nn.Linear(hidden_size, vocab_size)

        self.dropout = None
        if self.dropout_prob != 0:
            self.dropout = torch.nn.Dropout(self.dropout_prob)

    def zero_state(self, batch_size):
        state_shape = (batch_size, self.hidden_size)

        h0 = variable(torch.zeros(*state_shape))
        return h0

    def decoder_state(self, hidden):
        return hidden

    def decoder_initial_inputs(self, batch_size):
        inputs = variable(torch.from_numpy(np.full((1,), self.init_token, dtype=np.long)).expand((batch_size,)))
        return inputs

    def forward(self, hidden, targets=None):
        batch_size = hidden.size(0)

        if targets is not None:
            sequence_len = targets.size(1)
        else:
            sequence_len = self.max_len

        decoder_hidden = self.decoder_state(hidden)

        outputs = []
        decoder_inputs = self.decoder_initial_inputs(batch_size)
        for di in range(sequence_len):
            decoder_inputs = self.embedding(decoder_inputs)
            if self.dropout is not None:
                decoder_inputs = self.dropout(decoder_inputs)

            decoder_hidden = self.decoder(decoder_inputs, decoder_hidden)

            decoder_outputs = decoder_hidden
            if self.dropout is not None:
                decoder_outputs = self.dropout(decoder_outputs)

            out = self.out(decoder_outputs)
            outputs.append(out)

            if targets is not None:
                decoder_inputs = targets[:, di]
            else:
                decoder_inputs = argmax(out, dim=-1)

        outputs = torch.stack(outputs, dim=1)

        return outputs
