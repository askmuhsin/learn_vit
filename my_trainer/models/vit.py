import torch
import torch.nn as nn
import logging


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


class MLP(nn.Module):
    """
    Multi layer perceptron Layer
    
    Parameters:
        in_features: int, feature dim of input
        hidden_features: int, hidden layer features
        out_features: int, output layer features
        dropout_p: float, prob of dropout
        
    Learnables:
        fc1: nn.Linear
        fc2: nn.Linear
    """
    def __init__(
        self,
        in_features,
        hidden_features,
        out_features,
        dropout_p=.0,
    ):
        super().__init__()
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout_p)
    
    def forward(self, x):
        """
        Parameters:
            x: torch.Tensor, shape `(n_samples, n_tokens, emb)`
        
        Returns:
            x: torch.Tensor, shape `(n_samples, n_tokens, emb)`
        """
        x = self.act(self.fc1(x))  ## (n_samples, n_tokens, emb)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)           ## (n_samples, n_tokens, emb)
        
        return x


class Block(nn.Module):
    """
    The transformer block ties together the Attention and MLP layer.
    Before Attention and MLP, LayerNorm is applied as a regularizer.
    
    Parameters:
        dim: int, dimension of input patches after projection
        n_heads: int, number of heads in multi heads attention
        attn_p: float, dropout probability after attention layer
        proj_p: float, dropout probability after projection layer
        mlp_ratio: float, decides the size of mlp hidden layer
        mlp_p: float, dropout probability after mlp layer
        
    Learnables:
        norm_1: nn.LayerNorm
        norm_2: nn.LayerNorm
        
        attn: vit.Attention
        mlp: vit.MLP
    
    """
    def __init__(
        self,
        dim=768,
        n_heads=12,
        attn_p=.0,
        proj_p=.0,
        mlp_ratio=4.0,
        mlp_p=.0,
    ):
        super().__init__()
        
        self.norm_1 = nn.LayerNorm(dim, eps=1e-6)    
        self.attn = Attention(dim, n_heads, attn_p, proj_p)
        
        self.norm_2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLP(
            in_features=dim, 
            hidden_features=int(dim * mlp_ratio), 
            out_features=dim, 
            dropout_p=mlp_p
        )
    
    def forward(self, x):
        """
        Parameter:
            x: torch.Tensor, shape `(n_samples, n_tokens, emb)`
        
        Returns:
            torch.Tensor, shape `(n_samples, n_tokens, emb)`
        """
        x = x + self.attn(self.norm_1(x))
        x = x + self.mlp(self.norm_2(x))
        
        return x


class VisionTransformer(nn.Module):
    """
    The Vision Transformer block takes in an image, slices it up into patches,
    does a linear projection, prepends with the class token, adds the positional embedding,
    and puts it throught the enocder block, normalizes it, and projects to n_classes
    
    Parameters:
        image_channels: int, input channels of images
        patch_size: int, size of each square patch
        embedding_size: int, embedding size of projection of each patch
        n_heads: int, number of heads in multi heads attention
        attn_p: float, dropout probability after attention layer
        proj_p: float, dropout probability after projection layer
        mlp_ratio: float, decides the size of mlp hidden layer
        mlp_p: float, dropout probability after mlp layer
        encoder_depth: int, the number of sequential encoder blocks
        n_classes: int, number of classes in classifier
        pos_p: int, dropout probabilty for positional embedding
        

    Learnables:
        patch_embed: vit.PatchEmbeddings,
        cls_token: nn.Parameter, class token initilized to 0s
        positional_embeddings: nn.Parameter, positional embeddings initialized to 0s
        blocks: list[vit.Block], encoder_depth X encoder Blocks
        norm: nn.LayerNorm, layer normalization for final encoder output
        head: nn.Linear, final projection head to number of classes
    """
    def __init__(
        self,
        img_size,
        image_channels=3,
        patch_size=16,
        embedding_size=768,
        n_heads=12,
        attn_p=.0,
        proj_p=.0,
        mlp_ratio=4.0,
        mlp_p=.0,
        encoder_depth=12,
        n_classes=1000,
        pos_p=.0,
    ):
        super().__init__()

        self.n_patches = (img_size // patch_size) ** 2
        self.patch_embed = PatchEmbeddings(
            patch_size=patch_size,
            in_channels=image_channels,
            embedding_size=embedding_size,
        )
        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, embedding_size)
        )
        self.positional_embeddings = nn.Parameter(
            torch.zeros(1, 1 + self.n_patches, embedding_size)
        )
        self.pos_drop = nn.Dropout(p=pos_p)
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embedding_size,
                    n_heads=n_heads,
                    attn_p=attn_p,
                    proj_p=proj_p,
                    mlp_ratio=mlp_ratio,
                    mlp_p=mlp_p,
                )
                for _ in range(encoder_depth)
            ]
        )
        self.norm = nn.LayerNorm(embedding_size,  eps=1e-6)
        self.head = nn.Linear(embedding_size, n_classes)
    
    def forward(self, x):
        """
        Parameters:
            x: torch.Tensor, shape `(n_samples, n_channels, h, w)`
        
        Returns:
            torch.Tensor, shape `(n_samples, n_classes)`
        """
        n_samples = x.shape[0]
        logging.info(f'number of samples -- {n_samples}')
        x = self.patch_embed(x)  ## (n_samples, n_patches, embedding_size)
        logging.info(f'After patch embbedding -- {x.shape}')

        cls_tokens = self.cls_token.expand(n_samples, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  ## (n_samples, 1 + n_patches, embedding_size)
        x = x + self.positional_embeddings     ## (n_samples, 1 + n_patches, embedding_size)
        x = self.pos_drop(x)
        logging.info(f'After cls token and pos embedding -- {x.shape}')

        for block in self.blocks:
            x = block(x)                       ## (n_samples, 1 + n_patches, embedding_size)
        logging.info(f'After {len(self.blocks)} attention blocks -- {x.shape}')
        
        x = self.norm(x)

        cls_token_final = x[:, 0]              ## (n_samples, embedding_size)
        logging.info(f'Size of final class token -- {cls_token_final.shape}')

        x = self.head(cls_token_final)         ## (n_samples, class_size)
        logging.info(f'After final head -- {x.shape}')

        return x
