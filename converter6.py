# In converter.py
import torch
from flax import traverse_util
import torch
import jax
import jax.numpy as jnp
import numpy as np
from flax.serialization import to_bytes

# Import the JAX model and HyperParam class you created
from gcndesign_jax.models import GCNdesign, HyperParam
from gcndesign_jax.dataset import pdb2input_jax, add_margin_jax



def convert_and_save_weights(pytorch_path, jax_output_path):
    """
    Loads PyTorch weights, converts them to the JAX/Flax format,
    and saves them using a robust path-based mapping.
    """
    print("Starting weight conversion...")

    # 1. Load PyTorch state_dict
    loaded = torch.load(pytorch_path, map_location='cpu', weights_only=False)
    pt_state_dict = loaded.state_dict() if hasattr(loaded, 'state_dict') else loaded

    pt_params = {k: v for k, v in pt_state_dict.items() if 'running' not in k and 'num_batches_tracked' not in k}
    pt_batch_stats = {k: v for k, v in pt_state_dict.items() if 'running' in k}
    print(f"✅ Loaded PyTorch state_dict with {len(pt_params)} params and {len(pt_batch_stats)} batch_stats.")

    # 2. Initialize JAX model to get the target structure
    hypara = HyperParam()
    model = GCNdesign(hypara=hypara)
    key = jax.random.PRNGKey(0)
    
    node, edgemat, adjmat, label, mask, _ = pdb2input_jax("tmp.pdb", hypara)
    node_p, edgemat_p, adjmat_p, _, _ = add_margin_jax(node, edgemat, adjmat, label, mask, hypara.nneighbor)

    jax_variables = model.init({'params': key, 'dropout': key}, node_p, edgemat_p, adjmat_p, train=False)
    jax_params_scaffold = jax_variables['params']
    jax_batch_stats_scaffold = jax_variables.get('batch_stats', {})
    print("✅ Initialized JAX model to get parameter structure.")

    # 3. Map and Convert Weights (Robust Method)
    # ==========================================
    
    # Flatten both parameter trees to dictionaries with path-tuples as keys
    flat_jax_params = traverse_util.flatten_dict(jax_params_scaffold)
    flat_pt_params = {tuple(k.split('.')): v for k, v in pt_params.items()}

    # --- Convert Trainable Parameters ('params') ---
    assert len(flat_pt_params) == len(flat_jax_params), "Model parameter counts do not match!"
    
    # Create a mapping from PyTorch path tuples to JAX path tuples based on order
    pt_paths = list(flat_pt_params.keys())
    jax_paths = list(flat_jax_params.keys())
    path_mapping = {pt_path: jax_path for pt_path, jax_path in zip(pt_paths, jax_paths)}

    new_flat_jax_params = {}
    for pt_key_str, pt_tensor in pt_params.items():
        pt_path = tuple(pt_key_str.split('.'))
        jax_path = path_mapping[pt_path]

        np_tensor = pt_tensor.cpu().numpy()
        
        # Transpose weights based on the PyTorch key name
        if "weight" in pt_key_str:
            if len(np_tensor.shape) == 3: # Conv1D kernel
                np_tensor = np.transpose(np_tensor, (2, 1, 0))
            elif len(np_tensor.shape) == 2: # Dense kernel
                np_tensor = np_tensor.T
        
        new_flat_jax_params[jax_path] = np_tensor
        
    # Unflatten the dictionary to restore the nested structure
    new_jax_params = traverse_util.unflatten_dict(new_flat_jax_params)

    # --- Convert BatchNorm Statistics ('batch_stats') ---
    if pt_batch_stats:
        flat_jax_batch_stats = traverse_util.flatten_dict(jax_batch_stats_scaffold)
        flat_pt_batch_stats = {tuple(k.split('.')): v for k, v in pt_batch_stats.items()}
        
        assert len(flat_pt_batch_stats) == len(flat_jax_batch_stats), "BatchNorm layer counts do not match!"

        pt_bn_paths = list(flat_pt_batch_stats.keys())
        jax_bn_paths = list(flat_jax_batch_stats.keys())
        bn_path_mapping = {pt_path: jax_path for pt_path, jax_path in zip(pt_bn_paths, jax_bn_paths)}

        new_flat_jax_batch_stats = {}
        for pt_key_str, pt_tensor in pt_batch_stats.items():
            pt_path = tuple(pt_key_str.split('.'))
            jax_path = bn_path_mapping[pt_path]
            new_flat_jax_batch_stats[jax_path] = pt_tensor.cpu().numpy()

        new_jax_batch_stats = traverse_util.unflatten_dict(new_flat_jax_batch_stats)
    else:
        new_jax_batch_stats = {}

    print("✅ Conversion logic complete.")

    # 4. Save the converted weights
    final_jax_variables = {'params': new_jax_params, 'batch_stats': new_jax_batch_stats}
    param_bytes = to_bytes(final_jax_variables)
    with open(jax_output_path, "wb") as f:
        f.write(param_bytes)

    print(f"🎉 Successfully converted and saved JAX weights to '{jax_output_path}'")



if __name__ == '__main__':
    # Define the input and output file paths
    PYTORCH_PARAM_PATH = './gcndesign/params/param_default.pkl'
    JAX_PARAM_PATH = './gcndesign_jax/params/param_default_jax.msgpack'
    
    convert_and_save_weights(PYTORCH_PARAM_PATH, JAX_PARAM_PATH)
