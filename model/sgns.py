import torch
import torch.nn as nn


class SkipGram(nn.Module):
    def __init__(self, V: int, embedding_dim=100):
        """
        :param V: the size of vocabulary
        :param embedding_dim: the number of dimensions of word vector
        """
        super(SkipGram, self).__init__()

        self.embedding_dim = embedding_dim
        self.in_embeddings = nn.Embedding(V, embedding_dim, sparse=True)
        self.out_embeddings = nn.Embedding(V, embedding_dim, sparse=True)
        self.reset_parameters()

    def reset_parameters(self):
        upper = 0.5 / self.embedding_dim
        self.in_embeddings.weight.data.uniform_(-upper, upper)
        self.out_embeddings.weight.data.zero_()

    def forward(self, inputs, contexts, negatives):
        """
        :param inputs: (#mini_batches, 1)
        :param contexts: (#mini_batches, 1)
        :param negatives: (#mini_batches, #negatives)
        :return:
        """
        in_vectors = self.in_embeddings(inputs)
        pos_context_vectors = self.out_embeddings(contexts)
        neg_context_vectors = self.out_embeddings(negatives)

        pos = torch.sum(in_vectors * pos_context_vectors, dim=(1, 2))
        neg = torch.sum(in_vectors * neg_context_vectors, dim=2)

        return pos, neg
