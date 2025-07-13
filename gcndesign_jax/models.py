import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
from .hypara import HyperParam

from flax import linen as nn
import jax.numpy as jnp
from typing import Any

class InstanceNorm(nn.Module):
    epsilon: float = 1e-5
    use_scale: bool = True
    use_bias: bool = True
    param_dtype: Any = jnp.float32  # <<< added dtype
    dtype: Any = jnp.float32
    @nn.compact
    def __call__(self, x):
        x = jnp.asarray(x, self.dtype)

        mean = jnp.mean(x, axis=tuple(range(1, x.ndim - 1)), keepdims=True)
        var = jnp.var(x, axis=tuple(range(1, x.ndim - 1)), keepdims=True)
        x_norm = (x - mean) / jnp.sqrt(var + self.epsilon)

        if self.use_scale:
            scale = self.param('scale', nn.initializers.ones, (x.shape[-1],), self.dtype)
            x_norm *= scale.reshape((1,) * (x.ndim - 1) + (-1,))
        if self.use_bias:
            bias = self.param('bias', nn.initializers.zeros, (x.shape[-1],), self.dtype)
            x_norm += bias.reshape((1,) * (x.ndim - 1) + (-1,))

        return x_norm


## ResBlock with InstanceNormalization
class ResBlock_InstanceNorm(nn.Module):
    d_in: int
    d_out: int
    dropout_rate: float = 0.2

    @nn.compact
    def __call__(self, x, train: bool):
        x_bn1 = InstanceNorm(param_dtype=jnp.float32, name='bn1')(x)
        x_relu1 = nn.relu(x_bn1)
        x_conv1 = nn.Conv(features=self.d_out, kernel_size=(1,), strides=(1,), padding='SAME',
                          name='conv1', kernel_init=nn.initializers.kaiming_normal(), bias_init=nn.initializers.zeros)(x_relu1)
        x_bn2 = InstanceNorm(param_dtype=jnp.float32, name='bn2')(x_conv1)
        x_relu2 = nn.relu(x_bn2)
        x_dropout2 = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x_relu2)
        x_conv2 = nn.Conv(features=self.d_out, kernel_size=(1,), strides=(1,), padding='SAME',
                          name='conv2', kernel_init=nn.initializers.kaiming_normal(), bias_init=nn.initializers.zeros)(x_dropout2)
        shortcut = x
        if self.d_in != self.d_out:
            shortcut = InstanceNorm(param_dtype=jnp.float32, name='shortcut_bn')(shortcut)
            shortcut = nn.Conv(features=self.d_out, kernel_size=(1,), strides=(1,), padding='SAME',
                               name='shortcut_conv', kernel_init=nn.initializers.kaiming_normal(), bias_init=nn.initializers.zeros)(shortcut)
        return x_conv2 + shortcut

## ResBlock with BatchNormalization
class ResBlock_BatchNorm(nn.Module):
    d_in: int
    d_out: int
    dropout_rate: float = 0.2

    @nn.compact
    def __call__(self, x, train: bool):
        x_bn1 = nn.BatchNorm(param_dtype=jnp.float32, name='bn1', use_running_average=not train)(x)
        x_relu1 = nn.relu(x_bn1)
        x_conv1 = nn.Conv(features=self.d_out, kernel_size=(1,), strides=(1,), padding='SAME',
                          name='conv1', kernel_init=nn.initializers.kaiming_normal(), bias_init=nn.initializers.zeros)(x_relu1)
        x_bn2 = nn.BatchNorm(param_dtype=jnp.float32, name='bn2', use_running_average=not train)(x_conv1)
        x_relu2 = nn.relu(x_bn2)
        x_dropout2 = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x_relu2)
        x_conv2 = nn.Conv(features=self.d_out, kernel_size=(1,), strides=(1,), padding='SAME',
                          name='conv2', kernel_init=nn.initializers.kaiming_normal(), bias_init=nn.initializers.zeros)(x_dropout2)
        shortcut = x
        if self.d_in != self.d_out:
            shortcut = nn.BatchNorm(param_dtype=jnp.float32, name='shortcut_bn', use_running_average=not train)(shortcut)
            shortcut = nn.Conv(features=self.d_out, kernel_size=(1,), strides=(1,), padding='SAME',
                               name='shortcut_conv', kernel_init=nn.initializers.kaiming_normal(), bias_init=nn.initializers.zeros)(shortcut)
        return x_conv2 + shortcut

class RGCBlock(nn.Module):
    d_in: int
    d_out: int
    d_edge_in: int
    d_edge_out: int
    nneighbor: int
    d_hidden_node: int
    d_hidden_edge: int
    nlayer_node: int
    nlayer_edge: int
    dropout: float

    @nn.compact
    def __call__(self, x, edgevec, adjmat, training: bool):
        print(edgevec.shape)
        L, d_in = x.shape
        k_node = self.d_out - self.d_in
        k_edge = self.d_edge_out - self.d_edge_in
        adj = adjmat.squeeze(-1)
        adj_indices = jnp.argsort(~adj, axis=1)[:, :self.nneighbor]
        #nodetrg = jnp.take(x, adjmat, axis=0)  # shape: (L, nneighbor, d_in)
        nodetrg = jnp.take(x, adj_indices, axis=0)
        print(nodetrg.shape)
        #(166, 166, 1, 20)
        nodesrc = jnp.broadcast_to(x[:, None, :], (x.shape[0], self.nneighbor, self.d_in))
        # 2. Broadcast source node features (same shape as nodetrg)
        #selfnode = jnp.broadcast_to(x[:, None, :], (x.shape[0], adjmat.shape[1], x.shape[1]))  # (L, nneighbor, d_in)
        selfnode = jnp.broadcast_to(x[:, None, :], (L, self.nneighbor, d_in))  # (L, nneighbor, d_in)
        # 3. Concatenate [source, edge, target]
        # 4. Transpose to match PyTorch's Conv1d input: (L, d_total, nneighbor)
        nodetrg = nodetrg.squeeze()
        if self.nlayer_edge > 0 and k_edge > 0:
            nen = jnp.concatenate([selfnode, edgevec, nodetrg], axis=-1)
            #nen = jnp.transpose(nen, (0, 2, 1))  # (L, d_total, nneighbor)
            #print(jnp.transpose(nen, (0, 1, 2) )[0][0])
            nen_out = nn.Conv(features=self.d_hidden_edge, kernel_size=(1,), padding='SAME')(nen)

            for _ in range(self.nlayer_edge):
                nen_out = ResBlock_BatchNorm(d_in=self.d_hidden_edge, d_out=self.d_hidden_edge,
                                             dropout_rate=self.dropout)(nen_out, training)

            #print(nen_out)
            nen_out = ResBlock_BatchNorm(d_in=self.d_hidden_edge, d_out=k_edge,
                                         dropout_rate=self.dropout)(nen_out, training)

            nen_out = nn.BatchNorm(use_running_average=not training)(nen_out)
            nen_out = nn.relu(nen_out)
            edgevec = jnp.concatenate([edgevec, nen_out], axis=-1)

        nodeedge = jnp.concatenate([nodesrc, edgevec, nodetrg], axis=-1)
        encoded = nn.Conv(features=self.d_hidden_node, kernel_size=(1,), padding='SAME')(nodeedge)
        for _ in range(self.nlayer_node):
            encoded = ResBlock_BatchNorm(d_in=self.d_hidden_node, d_out=self.d_hidden_node,
                                         dropout_rate=self.dropout)(encoded, training)
        encoded = nn.BatchNorm(use_running_average=not training)(encoded)
        encoded = nn.relu(encoded)
        aggregated = jnp.sum(encoded, axis=1)
        residual = aggregated
        for _ in range(self.nlayer_node):
            residual = ResBlock_BatchNorm(d_in=self.d_hidden_node, d_out=self.d_hidden_node,
                                          dropout_rate=self.dropout)(residual[:, None, :], training).squeeze(1)
        residual = ResBlock_BatchNorm(d_in=self.d_hidden_node, d_out=k_node,
                                      dropout_rate=self.dropout)(residual[:, None, :], training).squeeze(1)
        residual = nn.BatchNorm(use_running_average=not training)(residual)
        residual = nn.relu(residual)
        out = jnp.concatenate([x, residual], axis=-1)
        return out, edgevec


class Embedding_module(nn.Module):
    nneighbor: int
    r_drop: float
    d_node0: int
    d_hidden_node0: int
    nlayer_node0: int
    d_hidden_node: int
    d_hidden_edge: int
    nlayer_node: int
    nlayer_edge: int
    niter_rgc: int
    k_node_rgc: int
    k_edge_rgc: int
    fragment_size: int
    d_node_in: int = 6
    d_edge_in: int = 36

    @nn.compact
    def __call__(self, node_in, edgemat_in, adjmat_in, train: bool):
        naa = node_in.shape[0]
        kernel_size = (self.fragment_size - 1) // 2 + 1

        adj = adjmat_in.squeeze(-1)  # (L, L) boolean
        L = edgemat_in.shape[0]
        d_edge = edgemat_in.shape[-1]
        
        print("adj shape:", adj.shape)
        print("adj dtype:", adj.dtype)
        print("True counts per row:", adj.sum(axis=1))
        # Sanity check
        edge_flat = edgemat_in[adj]  # shape (L * nneighbor, d_edge)
        print("edge_flat shape:", edge_flat.shape)
        edge = edge_flat.reshape(L, self.nneighbor, d_edge)
        #edge = edge_flat.reshape(L, -1, d_edge)


        # initial conv: (1, L, d_node_in)
        node = node_in[None, ...]

        #node = jnp.transpose(node, (1, 0))

        node = nn.Conv(features=self.d_hidden_node0, kernel_size=(kernel_size,), padding='SAME',
                       kernel_init=nn.initializers.kaiming_normal(), bias_init=nn.initializers.zeros)(node)



        #node = jnp.transpose(node, (1, 0))
        for _ in range(self.nlayer_node0):
            node = ResBlock_InstanceNorm(d_in=self.d_hidden_node0, d_out=self.d_hidden_node0,
                                      dropout_rate=self.r_drop)(node, train)
        #    print(node)
        #    break
        #    prin("d",node.shape)
        node = ResBlock_InstanceNorm(d_in=self.d_hidden_node0, d_out=self.d_node0,
                                     dropout_rate=self.r_drop)(node, train)



        node = InstanceNorm(param_dtype=jnp.float32)(node)
        node = nn.relu(node)
        node = node.squeeze()
       # print("e",node.shape)
        # GCN iterations
        for i in range(self.niter_rgc):
            node, edge = RGCBlock(
                d_in=self.d_node0 + self.k_node_rgc * i,
                d_out=self.d_node0 + self.k_node_rgc * (i + 1),
                d_edge_in=self.d_edge_in + (self.k_edge_rgc if self.nlayer_edge>0 else 0) * i,
                d_edge_out=self.d_edge_in + (self.k_edge_rgc if self.nlayer_edge>0 else 0) * (i + 1),
                nneighbor=self.nneighbor,
                d_hidden_node=self.d_hidden_node,
                d_hidden_edge=self.d_hidden_edge,
                nlayer_node=self.nlayer_node,
                nlayer_edge=self.nlayer_edge,
                dropout=self.r_drop
            )(node, edge, adjmat_in, train)


        return node, edge

class Prediction_module(nn.Module):
    d_in: int
    d_out: int
    d_hidden1: int
    d_hidden2: int
    nlayer_pred: int
    fragment_size: int
    r_drop: float

    @nn.compact
    def __call__(self, node_in, train: bool):
        kernel_size = (self.fragment_size - 1) // 2 + 1
        node = node_in[None, ...]
        node = nn.Conv(features=self.d_hidden1, kernel_size=(kernel_size,), padding='SAME',
                       kernel_init=nn.initializers.kaiming_normal(), bias_init=nn.initializers.zeros)(node)
        for _ in range(self.nlayer_pred):
            node = ResBlock_InstanceNorm(d_in=self.d_hidden1, d_out=self.d_hidden1,
                                         dropout_rate=self.r_drop)(node, train)
        node = InstanceNorm(param_dtype=jnp.float32)(node)
        node = nn.relu(node)
        node = nn.Conv(features=self.d_hidden2, kernel_size=(kernel_size,), padding='SAME',
                       kernel_init=nn.initializers.kaiming_normal(), bias_init=nn.initializers.zeros)(node)
        for _ in range(self.nlayer_pred):
            node = ResBlock_InstanceNorm(d_in=self.d_hidden2, d_out=self.d_hidden2,
                                         dropout_rate=self.r_drop)(node, train)
        node = InstanceNorm(param_dtype=jnp.float32)(node)
        node = nn.relu(node)
        node = nn.Conv(features=self.d_out, kernel_size=(1,), padding='SAME',
                       kernel_init=nn.initializers.kaiming_normal(), bias_init=nn.initializers.zeros)(node)
        return node[0]

class GCNdesign(nn.Module):
    hypara: HyperParam

    @nn.compact
    def __call__(self, node_in, edgemat_in, adjmat_in, train: bool):
        latent, _ = Embedding_module(
            nneighbor=self.hypara.nneighbor,
            r_drop=self.hypara.r_drop,
            d_node0=self.hypara.d_embed_node0,
            fragment_size=self.hypara.fragment_size0,
            d_hidden_node0=self.hypara.d_embed_h_node0,
            nlayer_node0=self.hypara.nlayer_embed_node0,
            d_hidden_node=self.hypara.d_embed_h_node,
            d_hidden_edge=self.hypara.d_embed_h_edge,
            nlayer_node=self.hypara.nlayer_embed_node,
            nlayer_edge=self.hypara.nlayer_embed_edge,
            niter_rgc=self.hypara.niter_embed_rgc,
            k_node_rgc=self.hypara.k_node_rgc,
            k_edge_rgc=self.hypara.k_edge_rgc
        )(node_in, edgemat_in, adjmat_in, train)
        out = Prediction_module(
            d_in=self.hypara.d_embed_node0 + self.hypara.k_node_rgc * self.hypara.niter_embed_rgc,
            d_out=self.hypara.d_pred_out,
            r_drop=self.hypara.r_drop,
            d_hidden1=self.hypara.d_pred_h1,
            d_hidden2=self.hypara.d_pred_h2,
            nlayer_pred=self.hypara.nlayer_pred,
            fragment_size=self.hypara.fragment_size
        )(latent, train)
        return out, latent

    def process_pdbfile(self, pdbfile, require_all=False):
        node, edgemat, adjmat, label, mask, res = pdb2input_jax(pdbfile, self.hypara)
        if require_all:
            return node, edgemat, adjmat, label, mask, res
        return node, edgemat, adjmat
