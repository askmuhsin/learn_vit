import torch
import torch.nn as nn


class PatchEmbeddings(nn.Module):
    """
    Input image is split into patches based on the patch_size.
    each patch is projected to embedding_size.
    the projection is reshaped
    
    input shape x -- (n_sample, n_channels, h, w)
    after projection -- (n_sample, embedding_size, n_p_h, n_p_w)
    after transformation -- (n_samples, n_patches, embedding_size)
    
    Parameters:
        patch_size: int, size of each square patch (default 16 for 16X16)
        in_channels: int, number of input channels (default 3 for RGB)
        embedding_size: int, the projection size (default 768)
    
    Learnables:
        projection: nn.Conv2d, a learnable layer of dim - 768 X [16 X 16 X 3]
        
    """
    def __init__(
        self,
        patch_size=16,
        in_channels=3,
        embedding_size=768,
    ):
        super().__init__()
        self.projection = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=embedding_size, 
            kernel_size=patch_size, 
            stride=patch_size,
        )
    
    def forward(self, x):
        """
        Parameters:
            x: torch.Tensor, shape `(n_samples, input_channels, h, w)`
        
        Returns:
            torch.Tensor, shape `(n_samples, n_patches, embedding_size)`
        """
        x = self.projection(x) ## (n_samples, embedding_size, n_p_h, n_p_w)
        x = x.flatten(2)       ## (n_samples, embedding_size, n_patches)
        x = x.transpose(1, 2)  ## (n_samples, n_patches, embedding_size)
        return x


class Attention(nn.Module):
    """
    applies the multi Attention mechanism to the input.
    the weights for q, k, v, for all the multi head attentions are initialized together.

    Parameters:
        dim: int, the input dim from patchEmbeddings (default 768)
        n_heads: int, number of multi head attention layers
        attn_p: float, probability for Dropout on attention weight
        proj_p: float, probability for Dropout on projection out
    
    Learnables:
        qkv: nn.Linear, q, k, v weights for all heads are initilized here
        proj: nn.Linear, a projection of same size as input after applying attention weights

    """
    def __init__(
        self,
        dim=768,
        n_heads=12,
        attn_p=.0,
        proj_p=.0,
    ):
        super().__init__()
        
        self.dim = dim
        self.n_heads = n_heads
        self.head_dims = dim // n_heads  ## defautls to 64 (dim - 768, n_heads - 12)
        self.scale = self.head_dims ** -.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_dropout = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_dropout = nn.Dropout(proj_p)
        
    def forward(self, x):
        """
        Parameters:
            x: torch.Tensor, shape `(n_samples, n_tokens, embedding_size)`

        Returns:
            torch.Tensor, shape `(n_samples, n_tokens, embedding_size)`
        """
        n_samples, n_tokens, dim = x.shape
        
        qkv = self.qkv(x)  ## (n_samples, n_tokens, embedding_size * 3)
        qkv = qkv.reshape(
            n_samples, n_tokens, 3, self.n_heads, self.head_dims
        ) ## (n_samples, n_tokens, 3, n_heads, head_dims)
        qkv = qkv.permute(2, 0, 3, 1, 4) ## (3, n_samples, n_heads, n_tokens, head_dims)
        
        q, k, v = qkv ## (n_samples, n_heads, n_tokens, head_dims)
        k_t = k.transpose(-1, -2) ## (n_samples, n_heads, head_dims, n_tokens)
        
        attn_weights = (q @ k_t) * self.scale  ## (n_samples, n_heads, n_tokens, n_tokens)
        attn_weights = attn_weights.softmax(dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        weighted_val = attn_weights @ v  ## (n_samples, n_heads, n_tokens, head_dims)
        weighted_val = weighted_val.transpose(1, 2)  ## (n_samples, n_tokens, n_heads, head_dims)
        weighted_val = weighted_val.flatten(2)  ## (n_samples, n_tokens, dim)
        
        x = self.proj(weighted_val)  ## (n_samples, n_tokens, dim)
        x = self.proj_dropout(x)
        
        return x
