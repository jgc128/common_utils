import numpy as np
import torch
import torch.nn.functional as F

from common_utils.torch.helpers import variable, argmax


class Decoder(torch.nn.Module):
    def __init__(self, embedding, hidden_size, max_len, init_token, nb_layers=1, dropout=0.2):
        super(Decoder, self).__init__()

        self.embedding = embedding
        self.hidden_size = hidden_size
        self.nb_layers = nb_layers
        self.dropout = dropout

        self.max_len = max_len
        self.init_token = init_token

        embedding_dim = embedding.weight.size(1)
        vocab_size = embedding.weight.size(0)
        self.decoder_cells = torch.nn.ModuleList([
            torch.nn.GRUCell(input_size=embedding_dim if i == 0 else hidden_size, hidden_size=hidden_size)
            for i in range(nb_layers)
        ])

        self.out = torch.nn.Linear(hidden_size, vocab_size)

    def zero_state(self, batch_size):
        state_shape = (batch_size, self.hidden_size)

        h0 = [variable(torch.zeros(*state_shape)) for _ in range(self.nb_layers)]
        return h0

    def decoder_state(self, hidden):
        return [hidden for _ in range(self.nb_layers)]

    def decoder_initial_inputs(self, batch_size):
        inputs = variable(torch.from_numpy(np.full((1,), self.init_token, dtype=np.long)).expand((batch_size,)))
        return inputs

    def _decoder_timestep(self, inputs, hidden):
        hidden_new = []
        for i, h_i in enumerate(hidden):
            if self.dropout != 0:
                inputs = F.dropout(inputs, p=self.dropout, training=self.training)

            h_i_new = self.decoder_cells[i](inputs, h_i)
            hidden_new.append(h_i_new)
            inputs = h_i_new

        return hidden_new

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

            # decoder_hidden = self.decoder(decoder_inputs, decoder_hidden)
            decoder_hidden = self._decoder_timestep(decoder_inputs, decoder_hidden)

            decoder_outputs = decoder_hidden[-1]
            if self.dropout != 0:
                decoder_outputs = F.dropout(decoder_outputs, p=self.dropout, training=self.training)

            out = self.out(decoder_outputs)
            outputs.append(out)

            if targets is not None:
                decoder_inputs = targets[:, di]
            else:
                decoder_inputs = argmax(out, dim=-1)

        outputs = torch.stack(outputs, dim=1)

        return outputs
