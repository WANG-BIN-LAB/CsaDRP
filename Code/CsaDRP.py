import torch.nn as nn
import numpy as np
import torch
from gensim.models import Word2Vec
import pandas as pd

import torch
import torch.nn.functional as F
import math


import warnings
warnings.filterwarnings("ignore")

from mamba_ssm.modules.mamba_simple import Mamba
import sys
sys.path.append('/data2/zyt/ResGitDR-main/models/')
from mamba2_simple import Mamba2Simple
#from native_sparse_attention_pytorch import SparseAttention
import torch.nn.utils.prune as prune

l2_reg_weight = 0.02



#MLP
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))

        for i in range(1, len(hidden_sizes)):
            self.layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x) 
        return x





class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )






##Mamba SGA
class SGA_Mamba(nn.Module):
    """
    This class implement Mamba for SGA.  Given a vector of SGA status of m genes, this class produced
    a vector of embedding_dim, which corresponding to the dimension of vector embedding representing each gene.
    """
    def __init__(self, sga_gene_list, sga_size, embedding_dim,n_head, using_gene2vec_pretrain=True, 
                 jamba_experts_per_token=1,sparse_attn_kwargs: dict = dict(
                     sliding_window_size = 32,
                     compress_block_size = 4,
                     compress_block_sliding_stride = 2,
                     selection_block_size = 4,
                     num_selected_blocks = 4,
                    
                 )):
        super(SGA_Mamba, self).__init__()
        self.sga_gene_list = sga_gene_list
        self.sga_size = sga_size
        self.embeddingDim = embedding_dim
        self.attention_head = n_head
        self.attention_size = embedding_dim
        self.using_gene2vec_pretrain = using_gene2vec_pretrain
        
        
       
       
        

        # set up a look up embedding matrix for SGAs, with first embedding vector being "0" at indx 0 as padding
        # A vector in this matrix corresponding to input vecotr (look up) as well as the v (identify function)
        if using_gene2vec_pretrain == True:
            print("using_gene2vec_pretrain")
            gene2vec_model = Word2Vec.load("/data2/zyt/ResGitDR-main/ResGitDR_data/data/data/gene2vec/word2vec.model_"+str(embedding_dim))
            w2v_dict = {item : gene2vec_model.wv[item] for item in gene2vec_model.wv.key_to_index}
            gene2vec_df = pd.DataFrame.from_dict(w2v_dict)
            gene2vec_df = gene2vec_df.loc[:,self.sga_gene_list]
            embedding_matrix = np.insert(gene2vec_df.values,0,np.zeros(gene2vec_df.shape[0]),axis=1)
            
            self.sga_embs = nn.Embedding.from_pretrained(torch.Tensor(embedding_matrix).T, padding_idx=0,freeze=False)
        else:
            self.sga_embs = nn.Embedding(num_embeddings=self.sga_size + 1, embedding_dim=int(self.embeddingDim),
                                         padding_idx=0) 
       
 
        
        self.Mamba = Mamba2Simple(d_model=self.embeddingDim, d_state=64, d_conv=4, expand=2,)
     
        """self.sparse_attention = SparseAttention(
                dim = embedding_dim,
                dim_head = embedding_dim // n_head,
                heads = n_head,
                kv_heads = None,
                causal = True,
                **sparse_attn_kwargs
            )"""
        

        self.Kan = KAN([self.sga_size, 256, 1])

    def forward(self, sga, mask=True):

        # Look up gene embedings based on input SGA data. Produce a m-by-embedding_dim matrix
        case_sga_embs = self.sga_embs(sga) # Batch x Input x Eembedding (e.g. 45 x 1084 x 200)
        #case_sga_embs = torch.randn(64, 1084, 200)

  
        
        


## Mamba
        emb_signal = self.Mamba(case_sga_embs)
        emb_signal =  emb_signal.permute(0, 2, 1)
        ## KAN
        batch = emb_signal.shape[0]
        embedding_dim = emb_signal.shape[1]
        sga_size = emb_signal.shape[2] 
        emb_signal = emb_signal.reshape(-1, sga_size)
        emb_signal = self.Kan(emb_signal)
        emb_signal = emb_signal.reshape(batch, embedding_dim, 1)
        emb_signal = emb_signal.reshape(emb_signal.shape[0],-1)
        return emb_signal, None




class CsaDRP(nn.Module):


  def __init__(self, out_feat_size, sga_size, embedding_dim, embedding_dim_last, can_size=10, n_head=5,
               num_attn_blocks=3,using_cancer_type=True,using_tf_gene_matrix=True,tf_gene=None,sga_gene_list=None,
               embed_pretrain_gene2vec=True,l2_reg_weight=l2_reg_weight):
    super(CsaDRP, self).__init__()
    self.sga_size = sga_size  #number of sga genes in total
    self.out_feat_size = out_feat_size #output embedding dimension
    self.can_size = can_size #number of cancer types in total
    self.embedding_dim = embedding_dim # embedding dimension
    self.embedding_dim_last = embedding_dim_last # embedding dimension for last layer
    self.attention_head = n_head #number of self attention heads
    self.activationF = nn.ReLU()
    self.num_attn_blocks = num_attn_blocks # number of sga attention blocks
    self.using_cancer_type = using_cancer_type #whether using cancer type information
    self.using_tf_gene_matrix = using_tf_gene_matrix #whether using tf-gene matrix
    self.tf_gene = tf_gene # the tf-gene matrix
    self.sga_gene_list = sga_gene_list #the sga gene list
    self.embed_pretrain_gene2vec = embed_pretrain_gene2vec #whether using gene2vec to pretrain the gene embedding

    self.gate_layers = nn.ModuleList()
    for i in range(num_attn_blocks - 1):  
            self.gate_layers.append(nn.Linear(embedding_dim, 1))
    self.cross_attn_layers = nn.ModuleList([
    nn.MultiheadAttention(embed_dim=self.embedding_dim, num_heads=4, batch_first=True)
    for _ in range(self.num_attn_blocks - 1)
    ])
    
    '''
        sga and can embeddings
    '''
    # cancer type embedding is a one-hot-vector, no need to pad
    self.layer_can_emb = nn.Embedding(num_embeddings=self.can_size, embedding_dim=self.embedding_dim)
    self.can_emb_layers = nn.ModuleList()
    for i in range (1, num_attn_blocks):
        self.can_emb_layers.append(nn.Embedding(num_embeddings=self.can_size, embedding_dim=self.embedding_dim))


    '''
        Construct the architecture of the GIT_RINN. 
    '''
    # we use two module lists: one to keep track of feed forward chain of hidden layers,
    # and one to keep track from SGA to hidden
    self.SGA_blocks = nn.ModuleList()
    self.hiddenLayers = nn.ModuleList()

    # First block is from SGA to hidden 0
    self.SGA_blocks.append(SGA_Mamba(self.sga_gene_list,self.sga_size, self.embedding_dim, self.attention_head, self.embed_pretrain_gene2vec))

    # populate the structure of hidden layers
    for i in range(1, num_attn_blocks-1):
        self.SGA_blocks.append(SGA_Mamba(self.sga_gene_list,self.sga_size, self.embedding_dim,self.attention_head, self.embed_pretrain_gene2vec))
        linearlayer = KAN([self.embedding_dim, self.embedding_dim])
        self.hiddenLayers.append(linearlayer)

    self.SGA_blocks.append(SGA_Mamba(self.sga_gene_list,self.sga_size, self.embedding_dim_last,self.attention_head,self.embed_pretrain_gene2vec))
    linearlayerLast = KAN([self.embedding_dim, self.embedding_dim_last])
    self.hiddenLayers.append(linearlayerLast)

    # final hidden to output layer
    if using_tf_gene_matrix:
        self.layer_final = nn.Linear(self.tf_gene.shape[0], self.out_feat_size,bias=False)
       # mask_value = torch.FloatTensor(self.tf_gene.T)
        self.mask_value = self.tf_gene.T

        # define layer weight clapped by mask
        self.layer_final.weight.data = self.layer_final.weight.data * torch.FloatTensor(self.tf_gene.T)
        # register a backford hook with the mask value
        self.layer_final.weight.register_hook(lambda grad: grad.mul_(self.mask_value))
        self.hiddenLayers.append(self.layer_final)
    else:
        # ??KAN?????TFgene
        #self.hiddenLayers.append(nn.Linear(self.embedding_dim_last, self.out_feat_size,bias=False))
        self.hiddenLayers.append(KAN(self.embedding_dim_last, self.out_feat_size))


  def forward(self, sga_index, can_index, store_hidden_layers=True):
    """ Forward process.
      Parameters
      ----------
      sga_index: list of sga index vectors.
      can_index: list of cancer type indices.
      -------
    """
    # attn_wts = [] # stores attention weights
    # First, feed forward from SGA to hidden 0
    e1, attn_wts = self.SGA_blocks[0](sga_index)
   
    if self.using_cancer_type:
        e1 = e1 + self.layer_can_emb(can_index)
        
        
     #   curr_hidden = curr_hidden + self.layer_can_emb(can_index).mean(dim=1).squeeze(1)
    curr_hidden = self.activationF(e1)

    batch_attn_wts = {}
    batch_attn_wts[0] = attn_wts # add attention weights obtained from first sga attention block

    hidden_layer_outs = {}
    hidden_layer_outs[0] = curr_hidden 



    for i in range(1, len(self.hiddenLayers)):
        #Cross Attention 
        sga_emb_after_attn, attn_wts = self.SGA_blocks[i](sga_index)
        if i < len(self.hiddenLayers) - 1:
           q = sga_emb_after_attn.unsqueeze(1)
           k = v = e1.unsqueeze(1)
           sga_emb_after_attn , _ = self.cross_attn_layers[i-1](q, k, v)
           sga_emb_after_attn = sga_emb_after_attn.squeeze(1)
        if self.using_cancer_type:
            #cancerembeeding ????
            curr_hidden = curr_hidden + self.can_emb_layers[i-1](can_index)
            
        
        # Resdual Gated Block
        gate = torch.sigmoid(self.gate_layers[i-1](curr_hidden))  # [batch, 1]
        transformed = self.hiddenLayers[i-1](curr_hidden)
        curr_hidden = gate * transformed + (1 - gate) * sga_emb_after_attn
        curr_hidden = self.activationF(curr_hidden) if i < len(self.hiddenLayers) - 1 else torch.sigmoid(curr_hidden)
        #attn_wts = attn_wts.detach().cpu().numpy() # store weights in numpy array

        if store_hidden_layers:
            hidden_layer_outs[i] = curr_hidden
        batch_attn_wts[i] = attn_wts

    preds = self.hiddenLayers[-1](curr_hidden)
    return preds, batch_attn_wts, hidden_layer_outs
            