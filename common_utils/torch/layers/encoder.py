import torch
import torch.nn.functional as F

from common_utils.torch.helpers import get_sequences_lengths
from common_utils.torch.layers.rnn_encoder import GRUEncoder


class Encoder(torch.nn.Module):
    def __init__(self, embedding, hidden_size, dropout=0.2, embedding_projection=False):
        super(Encoder, self).__init__()

        self.embedding = embedding
        self.dropout_prob = dropout

        embedding_dim = embedding.weight.size(1)

        self.emb_projection = None
        if embedding_projection:
            self.emb_projection = torch.nn.Linear(embedding_dim, embedding_dim)

        self.encoder = GRUEncoder(input_size=embedding_dim, hidden_size=hidden_size)

        self.dropout = None
        if self.dropout_prob != 0:
            self.dropout = torch.nn.Dropout(self.dropout_prob)

    def forward(self, inputs):
        inputs_lengths = get_sequences_lengths(inputs)

        encoder_inputs = self.embedding(inputs)
        if self.dropout is not None:
            encoder_inputs = self.dropout(encoder_inputs)

        if self.emb_projection is not None:
            encoder_inputs = F.elu(self.emb_projection(encoder_inputs))

            if self.dropout is not None:
                encoder_inputs = self.dropout(encoder_inputs)

        encoder_hidden = self.encoder(encoder_inputs, inputs_lengths)

        return encoder_hidden
