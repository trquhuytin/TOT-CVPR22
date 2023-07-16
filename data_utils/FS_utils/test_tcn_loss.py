import torch
from torch.nn import functional as F
import torch.nn as nn

class TCN(nn.Module):

    def __init__(self):
        super(TCN, self).__init__()

        
    def _npairs_loss(self, labels, embeddings_anchor, embeddings_positive):
        """Returns n-pairs metric loss."""
        
        # Get per pair similarities.
        similarity_matrix = torch.matmul(
            embeddings_anchor, embeddings_positive.t())

        # Reshape [batch_size] label tensor to a [batch_size, 1] label tensor.
        lshape = labels.shape

        # Add the softmax loss.
        xent_loss = F.cross_entropy(
            input=similarity_matrix, target=labels)
        #xent_loss = tf.reduce_mean(xent_loss)

        return xent_loss


    def single_sequence_loss(self, embs):
        """Returns n-pairs loss for a single sequence."""

        labels = torch.arange(embs.shape[0]/2).long()
        embeddings_anchor = embs[0::2]
        embeddings_positive = embs[1::2]
        loss = self._npairs_loss(labels, embeddings_anchor, embeddings_positive)
        return loss

    def forward(self, embs):
        return self.single_sequence_loss(embs)




x = torch.rand(256, 30)
tcn_loss = TCN()
print(tcn_loss(x))