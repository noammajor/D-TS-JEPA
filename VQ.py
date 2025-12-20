import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    """
    TOTEM-style Vector Quantizer for Time Series.
    Modified to handle 3D [Batch, Seq, Dim] inputs directly.
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1 / self._num_embeddings, 1 / self._num_embeddings)

    def forward(self, inputs):
        # Inputs shape: [Batch, Seq, Dim]
        input_shape = inputs.shape
        
        # Flatten input to [Batch * Seq, Dim]
        flat_input = inputs.reshape(-1, self._embedding_dim)

        # Calculate distances between input vectors and codebook embeddings
        # d = x^2 + y^2 - 2xy
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight ** 2, dim=1) 
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding: Find the nearest neighbor index
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize: Map indices back to codebook vectors
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Loss calculation (Commitment + Codebook loss)
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        # Straight-Through Estimator (STE)
        quantized = inputs + (quantized - inputs).detach()

        # Perplexity measures how many codes are being used
        avg_probs = torch.mean(encodings, dim=0)
        #track for the complexity of the model
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return loss, quantized, perplexity, encoding_indices