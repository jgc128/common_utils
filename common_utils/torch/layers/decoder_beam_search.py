import numpy as np
import torch
import torch.nn.functional as F

from common_utils.torch.helpers import variable
from common_utils.torch.layers.decoder import Decoder


class Beam(object):
    def __init__(self, hidden, inputs):
        self.sequence = []
        self.score = variable(torch.ones(1)).squeeze()

        self.hidden = hidden
        self.inputs = inputs

    def step(self, score, outputs, hidden):
        beam = Beam(hidden, outputs)

        beam.score = self.score + score
        beam.sequence = self.sequence + [int(outputs), ]

        return beam

    def __str__(self):
        sequence_str = ','.join(str(s) for s in self.sequence)
        return f'{sequence_str} - {self.score}'


class DecoderBeamSearch(Decoder):
    def __init__(self, beam_width=30, beam_sample=1000, *args, **kwargs):
        super(DecoderBeamSearch, self).__init__(*args, **kwargs)

        self.beam_width = beam_width
        self.beam_sample = beam_sample

    def beam_search(self, hidden):
        sequence_len = self.max_len

        beams = [
            Beam(hidden=self.decoder_state(hidden), inputs=self.decoder_initial_inputs(1).squeeze(0)),
        ]
        for di in range(sequence_len):
            beams_current = []

            decoder_hidden = torch.stack([b.hidden for b in beams])
            decoder_inputs = torch.stack([b.inputs for b in beams])
            beam_scores = torch.stack([b.score for b in beams])

            decoder_inputs = self.embedding(decoder_inputs)
            if self.dropout is not None:
                decoder_inputs = self.dropout(decoder_inputs)

            decoder_hidden = self.decoder(decoder_inputs, decoder_hidden)

            decoder_outputs = decoder_hidden
            if self.dropout is not None:
                decoder_outputs = self.dropout(decoder_outputs)

            out = self.out(decoder_outputs)
            out_probs = F.softmax(out, dim=-1)
            out_scores = F.log_softmax(out, dim=-1)

            candidates = torch.multinomial(out_probs, self.beam_sample)
            # candidates_scores = torch.index_select(out_scores, 0, candidates)
            candidates_scores = torch.gather(out_scores, 1, candidates)

            beams_current_scores = beam_scores.unsqueeze(1) + candidates_scores

            beams_current_scores = beams_current_scores.view(-1)

            beams_current_hidden_indices = variable(torch.arange(len(beams)))
            beams_current_hidden_indices = beams_current_hidden_indices.unsqueeze(1).expand(-1, self.beam_sample)
            beams_current_hidden_indices = beams_current_hidden_indices.contiguous().view(-1)

            beams_current_candidates = candidates.view(-1)

            top_scores, top_indices = torch.topk(beams_current_scores.view(-1), self.beam_width)
            top_hidden_indices = beams_current_hidden_indices[top_indices]
            top_candidates = beams_current_candidates[top_indices]

            for hidden_idx, candidate, score in zip(top_hidden_indices, top_candidates, top_scores):
                hidden_idx = int(hidden_idx)

                beam = beams[hidden_idx]
                beam_hidden = decoder_hidden[hidden_idx]

                beams_current.append(
                    beam.step(score, candidate, beam_hidden)
                )

            beams = beams_current

        outputs = variable(np.array(beams[0].sequence))

        return outputs
