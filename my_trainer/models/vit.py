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
    
    Attributes:
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
        x = self.projection(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x
