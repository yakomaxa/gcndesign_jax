### `training_jax.py`
import sys
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from flax.core import unfreeze
from typing import Any, Callable, Iterator, Tuple


# Assuming GCNdesign model and HyperParam class are in these paths
# from .hypara import HyperParam
# from .models import GCNdesign

# Define a custom TrainState to also manage batch normalization stats and RNG keys
class GCNTrainState(train_state.TrainState):
    batch_stats: Any
    key: jax.random.PRNGKey


# --- JAX-ported Helper Functions ---

def mat_connect_jax(mat1: jnp.ndarray, mat2: jnp.ndarray) -> jnp.ndarray:
    """
    Connects two 3D matrices (M, M, C) and (N, N, C) into a single
    block-diagonal matrix of shape (M+N, M+N, C).
    """
    print(mat1.shape)
    m_rows, _, channels = mat1.shape
    n_rows, _, _ = mat2.shape

    # Create zero-padding blocks to form the final matrix
    top_right = jnp.zeros((m_rows, n_rows, channels), dtype=mat1.dtype)
    bottom_left = jnp.zeros((n_rows, m_rows, channels), dtype=mat1.dtype)

    # Assemble the rows and concatenate them
    top_row = jnp.concatenate([mat1, top_right], axis=1)
    bottom_row = jnp.concatenate([bottom_left, mat2], axis=1)
    result = jnp.concatenate([top_row, bottom_row], axis=0)

    return result


def BatchLoaderJax(dataloader: Iterator, maxsize: int) -> Iterator[Tuple]:
    """
    A generator that dynamically accumulates and combines multiple small batches
    from a dataloader into larger batches that do not exceed a total size.
    This is a functional replacement for the original stateful BatchLoader class.

    Yields:
        A tuple containing the combined batch data:
        (node, edge, adj, target, mask, name, num_in_batch)
    """
    batch_cache = []
    current_size = 0
    #total_samples = len(dataloader)

    for item in dataloader:
        item_size = item[0].shape[1]  # Get sequence length from node features

        # If the cache has items and adding the next one exceeds the max size,
        # yield the currently cached batch.
        if batch_cache and current_size + item_size > maxsize:
            yield _combine_batches(batch_cache)
            batch_cache = []
            current_size = 0

        batch_cache.append(item)
        current_size += item_size

    # Yield the final remaining batch in the cache
    if batch_cache:
        yield _combine_batches(batch_cache)


def _combine_batches(batches: list) -> Tuple:
    """Helper function to combine a list of batches into a single batch."""
    if len(batches) == 1:
        node, edge, adj, target, mask, name = batches[0]
        # Ensure name is a plain string and return num=1
        return node, edge, adj, target, mask, str(name[0]), 1

    # Unzip the list of batch tuples into separate lists
    nodes, edges, adjs, targets, masks, names = zip(*batches)

    # Concatenate along the sequence length dimension (axis=1)
    final_node = jnp.concatenate(nodes, axis=0)
    final_target = jnp.concatenate(targets, axis=0)
    final_mask = jnp.concatenate(masks, axis=0)

    # Use mat_connect_jax iteratively to build the block-diagonal matrices
    # Squeeze and re-expand the batch dimension (which is 1)
    final_edge_3d = edges[0]#.squeeze(0)
    for e in edges[1:]:
        final_edge_3d = mat_connect_jax(final_edge_3d, e)#.squeeze(0))
    #final_edge = jnp.expand_dims(final_edge_3d, 0)
    final_edge = final_edge_3d

    final_adj_3d = adjs[0]
    if final_adj_3d.ndim == 2:
        final_adj_3d = final_adj_3d[..., None]

    if final_adj_3d.shape[0] != final_node.shape[0]:
        # Shape is flipped (e.g. (k, L, 1) instead of (L, k, 1))
        final_adj_3d = jnp.transpose(final_adj_3d, (1, 0, 2))
    print(final_adj_3d.shape)
    for a in adjs[1:]:
        if a.ndim == 2:
            a = a[..., None]
        if a.shape[0] != final_node.shape[0]:
            a = jnp.transpose(a, (1, 0, 2))
        final_adj_3d = mat_connect_jax(final_adj_3d, a)#.squeeze(0))
    #final_adj = jnp.expand_dims(final_adj_3d, 0)
    #final_adj = final_adj.squeeze(2)
    final_adj = final_adj_3d
    #hoge
    # Combine names and count the number of source PDBs in the batch
    final_name = '_'.join([str(n[0]) for n in names])
    num_in_batch = len(batches)

    print("Final adj shape:", final_adj.shape)
    print("Final edge shape:", final_edge.shape)
    print("Final node shape:", final_node.shape)
    return final_node, final_edge, final_adj, final_target, final_mask, final_name, num_in_batch
    #return final_node, final_edge, final_adj, final_target, final_mask, num_in_batch


def _preprocess_model_inputs(batch: Tuple) -> Tuple:
    """
    Prepares raw batch data for the model's apply function, primarily by
    flattening the edge features based on the adjacency matrix, which is
    required by the GCN model architecture.
    """
    node, edge, adj, target, mask, _ = batch

    # Squeeze the batch dimension (assumed to be 1)
    if node.ndim == 3 and node.shape[0] == 1:
        node_p = node.squeeze(0)
        edgemat_p = edge.squeeze(0)
        adjmat_p = adj.squeeze(0)
        target_p = target.squeeze(0)
        mask_p = mask.squeeze(0)
    else:
        node_p = node
        edgemat_p = edge
        adjmat_p = adj
        target_p = target
        mask_p = mask

    #adj_bool = adjmat_p[..., 0]

    # Get neighbor indices per node
    #neighbor_idx = jnp.argsort(-adj_bool, axis=1)[:, :nneighbor]
    # Assuming adjacency is boolean or 0/1, argsort(-adj_bool) puts True first

    # Gather edges for those neighbors: shape (L, nneighbor, d_edge)
    #edge_flat = jnp.take_along_axis(edgemat_p, neighbor_idx[:, :, None], axis=1)

    # Flatten to (L * nneighbor, d_edge)
    #edge_flat = edge_flat.reshape(-1, edgemat_p.shape[-1])
    adj_bool = adjmat_p.squeeze(-1)
    #print(adj_bool)
    #print(adj_bool.sum())
    edge_flat = edgemat_p[adj_bool]
    edge_flat = edge_flat[None,:]
    return node_p, edge_flat, adjmat_p, target_p, mask_p


from functools import partial

@partial(jax.jit, static_argnames=['criterion', 'param_path', 'freeze_embedding'])
def _train_step_jit(state: GCNTrainState,
                    node: jnp.ndarray,
                    edge_flat: jnp.ndarray,
                    adjmat: jnp.ndarray,
                    target: jnp.ndarray,
                    mask: jnp.ndarray,
                    criterion: Callable,
                    freeze_embedding: bool,
                    param_path: Tuple[str, ...]) -> Tuple:
    dropout_key, new_key = jax.random.split(state.key)

    def loss_fn(params):
        (logits, _), new_model_state = state.apply_fn(
            {'params': params, 'batch_stats': state.batch_stats},
            node, edge_flat, adjmat,
            train=True,
            mutable=['batch_stats'],
            rngs={'dropout': dropout_key}
        )
        # Use mask to compute loss and accuracy
        loss = jnp.sum(criterion(logits, target) * mask) / jnp.sum(mask)
        accuracy = jnp.sum((jnp.argmax(logits, axis=-1) == target) * mask) / jnp.sum(mask)
        return loss, (accuracy, new_model_state['batch_stats'])

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (accuracy, new_batch_stats)), grads = grad_fn(state.params)

    if freeze_embedding:
        grads = unfreeze(grads)
        grad_dict = grads
        for key in param_path[:-1]:
            grad_dict = grad_dict[key]
        grad_dict[param_path[-1]] = jax.tree_map(jnp.zeros_like, grad_dict[param_path[-1]])
        grads = grads  # refreeze if needed

    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=new_batch_stats, key=new_key)

    return state, loss, accuracy  # ✅ Exactly 3 values

def _train_step(state, batch, criterion, freeze_embedding, param_path):
    node_p, edge_flat, adjmat_p, target, mask = _preprocess_model_inputs(batch)
    return _train_step_jit(state, node_p, edge_flat, adjmat_p, target, mask,
                           criterion, freeze_embedding, param_path)

#@jax.jit
def _eval_step(state, batch, criterion):
    node_p, edge_flat, adjmat_p, target, mask = _preprocess_model_inputs(batch)
    print(edge_flat.shape)
    logits, _ = state.apply_fn(
        {'params': state.params, 'batch_stats': state.batch_stats},
        node_p, edge_flat, adjmat_p,
        train=False
    )
    #print(state)
    preds = jnp.argmax(logits, axis=-1)
    correct = (preds == target).astype(jnp.float32)

    loss_raw = criterion(logits, target)
    loss = jnp.sum(loss_raw * mask) / jnp.sum(mask)
    accuracy = jnp.sum(correct * mask) / jnp.sum(mask)
    count = jnp.sum(mask)

    return loss, accuracy, count


def train_jax(state: GCNTrainState, criterion: Callable, train_loader: Any, hypara: Any, source: Any) -> Tuple:
    total_loss, total_correct, total_count, total_sample_count = 0.0, 0.0, 0.0, 0

    freeze_embedding = source.onlypred
    embedding_param_path = ('params', 'Embed_0')  # adjust as needed

    train_loader = list(train_loader)  # avoid exhausting generator
    batch_loader = BatchLoaderJax(iter(train_loader), hypara.batchsize_cut)
    total_batches = len(train_loader)

    for batch_data in batch_loader:
        # Prepare inputs
        batch_jit = batch_data[:5] + batch_data[6:]  # drop name
        state, loss, accuracy = _train_step(state, batch_jit, criterion, freeze_embedding, embedding_param_path)

        count = jnp.sum(batch_data[4])  # mask
        total_loss += loss.item() * count
        total_correct += accuracy.item() * count
        total_count += count
        total_sample_count += batch_data[6]  # num_in_batch

        sys.stderr.write(f'\r\033[K[{total_sample_count}/{total_batches}]')
        sys.stderr.flush()

    avg_loss = total_loss / total_count
    avg_acc = 100 * total_correct / total_count
    print(f' T.Loss: {avg_loss:.3f},  T.Acc: {avg_acc:.3f}', file=sys.stderr, end='')
    return state, avg_loss, avg_acc



def valid_jax(state: GCNTrainState, criterion: Callable, valid_loader: Any, hypara: Any) -> Tuple:
    total_loss, total_correct, total_count, total_sample_count = 0.0, 0.0, 0.0, 0

    valid_loader = list(valid_loader)  # Avoid exhausting generator
    batch_loader = BatchLoaderJax(iter(valid_loader), hypara.batchsize_cut)
    total_batches = len(valid_loader)

    for batch_data in batch_loader:
        batch_jit = batch_data[:5] + batch_data[6:]
        # Unpack inputs if needed — _eval_step expects full batch tuple
        loss, accuracy, count = _eval_step(state, batch_jit, criterion)

        total_loss += loss.item() * count
        total_correct += accuracy.item() * count
        total_count += count
        total_sample_count += batch_data[6]  # num_in_batch

        sys.stderr.write(f'\r\033[K[Validation: {total_sample_count}/{total_batches}]')
        sys.stderr.flush()

    avg_loss = total_loss / total_count
    avg_acc = 100 * total_correct / total_count
    print(f' V.Loss: {avg_loss:.3f}, V.Acc: {avg_acc:.3f}', file=sys.stderr)
    return avg_loss, avg_acc


def test_jax(state: GCNTrainState, criterion: Callable, test_loader: Any, hypara: Any) -> Tuple:
    total_loss, total_correct, total_count, total_sample_count = 0.0, 0.0, 0.0, 0

    test_loader = list(test_loader)
    batch_loader = BatchLoaderJax(iter(test_loader), hypara.batchsize_cut)
    total_batches = len(test_loader)

    for batch_data in batch_loader:
        batch_jit = batch_data[:5] + batch_data[6:]
        name = batch_data[5]  # filename or identifier
        loss, accuracy, count = _eval_step(state, batch_jit, criterion)

        print(f'Loss={loss.item():7.4f}   Acc={accuracy.item() * 100:6.2f} %  : L={count:4d}  ({name})')

        total_loss += loss.item() * count
        total_correct += accuracy.item() * count
        total_count += count
        total_sample_count += batch_data[6]

        sys.stderr.write(f'\r\033[K[Test: {total_sample_count}/{total_batches}]')
        sys.stderr.flush()

    avg_loss = total_loss / total_count
    avg_acc = 100 * total_correct / total_count
    print()  # for newline after progress
    return avg_loss, avg_acc