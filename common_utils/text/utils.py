import numpy as np

from common_utils.io import load_pickle


def load_embeddings(filename, top_k=None):
    word_vectors = load_pickle(filename)

    if top_k is not None:
        top_k_tokens = list(word_vectors.keys())[:top_k]

        word_vectors = {t: word_vectors[t] for t in top_k_tokens}

    return word_vectors


def create_embeddings_matrix(word_vectors, vocab, pad_token='<pad>'):
    emb_dim = len(next(iter(word_vectors.values())))
    vocab_size = len(vocab)

    W_emb = np.zeros((vocab_size, emb_dim))

    special_tokens = {t: np.random.uniform(-0.3, 0.3, (emb_dim,)) for t in vocab.special_tokens}
    special_tokens[pad_token] = np.zeros((emb_dim,))

    nb_unk = 0
    for i, t in vocab.id2token.items():
        if t in special_tokens:
            W_emb[i] = special_tokens[t]
        else:
            if t in word_vectors:
                W_emb[i] = word_vectors[t]
            else:
                W_emb[i] = np.random.uniform(-0.3, 0.3, emb_dim)
                nb_unk += 1

    print(f'Unknown tokens: {nb_unk}')
    return W_emb
