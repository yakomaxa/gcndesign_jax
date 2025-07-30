#! /usr/bin/env python

import sys
import argparse
from os import path
import pickle

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state, checkpoints
from flax import serialization

# --- Add project root to path ---
# Assumes the script is run from a location where this relative path is valid.
dir_script = path.dirname(path.realpath(__file__))
sys.path.append(dir_script + '/../')

# --- Import JAX-ported modules ---
from gcndesign_jax.hypara import HyperParam, InputSource
from gcndesign_jax.dataset import BBGDatasetJAX  # Assumes a JAX-ported dataset
from gcndesign_jax.training import train_jax, valid_jax, GCNTrainState
from gcndesign_jax.models import GCNdesign  # Assumes a Flax GCNdesign model


def main():
    # --- Default Hyperparameters and Config ---
    hypara = HyperParam()
    source = InputSource()

    # --- Argument Parser (largely unchanged) ---
    parser = argparse.ArgumentParser(description="GCN-Design Training Script for JAX/Flax")
    # File I/O arguments
    parser.add_argument('--train_list', '-t', type=str, required=True, help='List of training data PDB files.')
    parser.add_argument('--valid_list', '-v', type=str, required=True, help='List of validation data PDB files.')
    parser.add_argument('--param_prefix', '-p', type=str, default=source.param_prefix,
                        help=f'Prefix for trained parameter output files. (default: "{source.param_prefix}")')
    parser.add_argument('--param_in', type=str, default=None,
                        help='Path to pre-trained parameter file for transfer learning.')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/',
                        help='Directory to save/load checkpoints. (default: ./checkpoints/)')
    parser.add_argument('--output', '-o', type=str, default=source.file_out,
                        help=f'Output log file. (default: "{source.file_out}")')

    # Training control arguments
    parser.add_argument('--epochs', '-e', type=int, default=hypara.nepoch,
                        help=f'Number of training epochs. (default: {hypara.nepoch})')
    parser.add_argument('--learning_rate', '-lr', type=float, default=hypara.learning_rate,
                        help=f'Learning rate. (default: {hypara.learning_rate})')
    parser.add_argument('--only_predmodule', action='store_true',
                        help='Freeze embedding layers and only train the prediction module (for transfer learning).')

    # Model architecture arguments (defaults are loaded from HyperParam)
    parser.add_argument('--dim_hidden_node', '-dn', type=int, default=hypara.d_embed_h_node,
                        help=f'Hidden dimensions of node-embedding layers. (default: {hypara.d_embed_h_node})')
    parser.add_argument('--layer_embed_node', '-ln', type=int, default=hypara.nlayer_embed_node,
                        help=f'Number of node-embedding layers. (default: {hypara.nlayer_embed_node})')
    # ... other architecture arguments can be added here in the same fashion ...

    args = parser.parse_args()

    # --- Update Hyperparameters from Args ---
    hypara.nepoch = args.epochs
    hypara.learning_rate = args.learning_rate
    hypara.d_embed_h_node = args.dim_hidden_node
    hypara.nlayer_embed_node = args.layer_embed_node
    # ... update other hypara fields ...

    source.file_train = args.train_list
    source.file_valid = args.valid_list
    source.onlypred = args.only_predmodule
    source.param_prefix = args.param_prefix
    source.file_out = args.output
    source.param_in = args.param_in

    # --- JAX Initialization ---
    key = jax.random.PRNGKey(0)
    key, init_key = jax.random.split(key)
    model = GCNdesign(hypara=hypara)

    # --- Model and State Initialization ---
    # Check if a pre-trained model file is provided for fine-tuning.
    # This uses the same direct loading method as your PredictorJax class.

    # Default is no --param_in is given, or if --param_in is given but the file doesn't exist
    if args.param_in and path.isfile(args.param_in):
        print(f"✅ Initializing model by loading variables from: {args.param_in}")
        with open(args.param_in, 'rb') as f:
            # Load the complete variables dict (params and batch_stats) directly.
            # The 'None' target tells Flax to reconstruct the structure from the file.
            variables = serialization.from_bytes(None, f.read())

    # If not loading, initialize new random parameters from scratch.
    else:
        if args.param_in:
            print(f"⚠️ Warning: Pre-trained model file not found at '{args.param_in}'.")
        print("✅ Initializing model with new random weights.")

        # Create dummy inputs to infer shapes for initialization.
        N, K = 32, hypara.nneighbor
        node_shape = (1, N, 6)
        edge_shape = (1, N, N, 36)
        adj_shape = (1, N, K, 1)  # Note: The model's init needs the unprocessed shapes.

        dummy_node = jnp.ones(node_shape, dtype=jnp.float32)
        dummy_edge = jnp.ones(edge_shape, dtype=jnp.float32)
        dummy_adj = jnp.ones(adj_shape, dtype=jnp.bool_)

        # model.init() creates the initial 'variables' dictionary.
        variables = model.init(
            {'params': init_key, 'dropout': init_key},
            dummy_node, dummy_edge, dummy_adj,
            train=False
        )

    # --- Optimizer and Scheduler Setup ---
    # This part remains the same.
    lr_schedule = optax.piecewise_constant_schedule(
        init_value=hypara.learning_rate,
        boundaries_and_scales={hypara.nepoch - 10: 0.1}
    )
    optimizer = optax.adam(learning_rate=lr_schedule)

    # --- Create the Training State ---
    # The 'variables' dict (either loaded or newly initialized) is used here.
    state = GCNTrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        batch_stats=variables['batch_stats'],
        tx=optimizer,
        key=key
    )

    # --- Restore from Checkpoint (if available) ---
    # This step comes last. It will overwrite the state created above if a
    # checkpoint from a previous run exists, which is perfect for resuming.
    state = checkpoints.restore_checkpoint(args.checkpoint_dir, target=state)
    epoch_init = int(state.step) + 1

    print(f"▶️ Starting training from epoch: {epoch_init}")

    # --- Handle Transfer Learning ---
    if source.onlypred and source.param_in:
        print(f"Performing transfer learning from: {source.param_in}")
        assert path.isfile(source.param_in), f"Parameter file {source.param_in} not found."
        with open(source.param_in, 'rb') as f:
            # Load pre-trained params and replace the current ones
            loaded_params = serialization.from_bytes(state.params, f.read())
            state = state.replace(params=loaded_params)
            # The freezing logic is handled inside `train_jax` via the `source.onlypred` flag.

    # --- Restore from Checkpoint if available ---
    # This will overwrite the initialized state if a checkpoint exists.
    state = checkpoints.restore_checkpoint(args.checkpoint_dir, target=state)
    epoch_init = int(state.step) + 1

    print(f"Starting training from epoch: {epoch_init}")

    # --- Dataloader Setup ---
    # We assume BBGDatasetJax yields JAX arrays and works with a Python generator.
    # The original uses shuffle=True, so we manually shuffle indices.
    def create_data_iterator(dataset_path, hypara):
        dataset = BBGDatasetJAX(listfile=dataset_path, hypara=hypara)
        indices = jnp.arange(len(dataset))
        shuffled_indices = jax.random.permutation(key, indices)

        for idx in shuffled_indices:
            yield dataset[idx]

    train_loader = lambda: create_data_iterator(source.file_train, hypara)
    valid_loader = lambda: create_data_iterator(source.file_valid, hypara)

    # --- Loss Function ---
    # The loss is calculated on logits, so `from_logits=True`.
    criterion = optax.softmax_cross_entropy_with_integer_labels

    # --- Training Routine ---
    param_count = sum(x.size for x in jax.tree_util.tree_flatten(state.params)[0])
    print(f"# Total Parameters: {param_count / 1_000_000:.2f}M")

    with open(source.file_out, 'w') as f_out:
        f_out.write(f"# Total Parameters : {param_count / 1_000_000:.2f}M\n")

        for iepoch in range(epoch_init, hypara.nepoch + 1):
            print(f"\n--- Epoch {iepoch}/{hypara.nepoch} ---")

            # Training
            state, loss_train, acc_train = train_jax(state, criterion, train_loader(), hypara, source)

            # Validation
            loss_valid, acc_valid = valid_jax(state, criterion, valid_loader(), hypara)

            # Log results
            log_line = (f' {iepoch:3d}  LossTR: {loss_train:.3f} AccTR: {acc_train:.3f}'
                        f'  LossVL: {loss_valid:.3f} AccVL: {acc_valid:.3f}\n')
            f_out.write(log_line)
            f_out.flush()
            print(log_line.strip())

            # Save model parameters (like the original .pkl) and a full checkpoint
            # 1. Save parameters only
            param_bytes = serialization.to_bytes(state.params)
            with open(f"{source.param_prefix}-{iepoch:03d}.msgpack", "wb") as f:
                f.write(param_bytes)

            # 2. Save full checkpoint for resuming training
            checkpoints.save_checkpoint(args.checkpoint_dir, target=state, step=iepoch, overwrite=True)

    print("\nTraining finished.")


if __name__ == "__main__":
    main()