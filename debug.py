import numpy as np
import torch
import jax
import jax.numpy as jnp

# Import PyTorch modules
from gcndesign.predictor import Predictor as PredictorTorch
from gcndesign.dataset import pdb2input, add_margin

# Import JAX modules
from gcndesign_jax.predictor import PredictorJax
from gcndesign_jax.dataset import pdb2input_jax, add_margin_jax

def print_comparison_results(name, arr_torch, arr_jax):
    """A helper function to print comparison stats."""
    print(f"\n--- Comparing: {name} ---")
    # Ensure inputs are NumPy arrays for comparison
    arr_torch = arr_torch.detach().cpu().numpy() if isinstance(arr_torch, torch.Tensor) else np.array(arr_torch)
    arr_jax = np.array(arr_jax)

    if arr_torch.shape != arr_jax.shape:
        print(f"❌ SHAPE MISMATCH: Torch: {arr_torch.shape}, JAX: {arr_jax.shape}")
        return

    # Check for perfect (bit-for-bit) equality
    if np.array_equal(arr_torch, arr_jax):
        print(f"✅ Perfect Match! (bit-for-bit identical)")
        return

    # If not perfect, check for numerical closeness
    if np.allclose(arr_torch, arr_jax, atol=1e-7):
        print(np.abs(arr_torch - arr_jax))
        print(f"✅ Numerically Close. Max absolute difference: {max_diff:.6e}")
    else:
        for i in range(3):
            print("torch",arr_torch[i])
            print("Jax--",arr_jax[i])
            print("diff",arr_jax[i]-arr_torch[i])
        max_diff = np.max(np.abs(arr_torch - arr_jax))
        print(f"❌ DIVERGENCE FOUND! Max absolute difference: {max_diff:.6e}")

def debug_models(pdb_file: str):
    """Performs a step-by-step comparison of the PyTorch and JAX models."""

    # --- 1. SETUP AND PREPARE INPUTS ---
    print("--- 1. Preparing Inputs ---")
    
    # PyTorch Inputs
    predictor_torch = PredictorTorch(device='cpu')
    node_t, edge_t, adj_t, lbl_t, msk_t, _ = pdb2input(pdb_file, predictor_torch.hypara)
    node_p_t, edge_p_t, adj_p_t, lbl_p_t, msk_p_t= add_margin(node_t, edge_t, adj_t, lbl_t, msk_t, predictor_torch.hypara.nneighbor)
    # Get PyTorch embedding
    predictor_torch.model.eval()
    latent_t, _ = predictor_torch.model.get_embedding(
        node_p_t.squeeze().to('cpu'), 
        edge_p_t.squeeze().to('cpu'), 
        adj_p_t.squeeze().to('cpu')
    )


    print(predictor_torch.hypara.nneighbor)
    # JAX Inputs
    predictor_jax = PredictorJax()
    node_j, edge_j, adj_j, lbl_j, msk_j, _ = pdb2input_jax(pdb_file, predictor_jax.hypara)
    node_p_j, edge_p_j, adj_p_j, lbl_p_j, msk_p_j = add_margin_jax(node_j, edge_j, adj_j, lbl_j, msk_j, predictor_jax.hypara.nneighbor)
    print(predictor_jax.hypara.nneighbor)

    # Compare the actual input tensors
    print_comparison_results("Input Node Features", node_p_t.squeeze(), node_p_j)
    print_comparison_results("Input Edge Features", edge_p_t.squeeze(), edge_p_j)
    print_comparison_results("Input Adj Features", adj_p_t.squeeze(), adj_p_j)
    print_comparison_results("Input lbl Features", lbl_p_t.squeeze(), lbl_p_j)
    print_comparison_results("Input msk Features", msk_p_t.squeeze(), msk_p_j)
    
        
    # Get JAX embedding
    #latent_j, _ = predictor_jax.model.get_embedding(node_p_j, edge_p_j, adj_p_j, train=False)
    
    
        
    # Get PyTorch logits
    logits_t, _ = predictor_torch._pred_base(pdb_file)
    
    # Get JAX logits
    logits_j, _ , latent_j = predictor_jax._pred_base(pdb_file)
    logits_j = logits_j[1:-1]
    print_comparison_results("Embedding Module Output (Latent)", latent_t, latent_j)
    #hoge
    #logits_j, _ = predictor_jax._pred_base(pdb_file)
    # --- 2. COMPARE INTERMEDIATE EMBEDDING ---
    print("\n--- 2. Comparing Intermediate Embedding Output ---")

    print_comparison_results("Input Adjacency Matrix", adj_p_t.squeeze(), adj_p_j)

    # --- 3. COMPARE FINAL LOGITS (PRE-SOFTMAX) ---
    print("\n--- 3. Comparing Final Logits (before Softmax) ---")

    
    print_comparison_results("Final Logits", logits_t, logits_j)
    
if __name__ == '__main__':
    PDB_TO_TEST = "tmp.pdb"
    debug_models(pdb_file=PDB_TO_TEST)
