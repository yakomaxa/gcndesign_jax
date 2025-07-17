import os
from os import path
import numpy as np
import jax
import jax.numpy as jnp
from flax import serialization
from flax.serialization import from_bytes
from flax.serialization import to_bytes

# Assuming these JAX-ported modules are in the specified paths
from .hypara import HyperParam
from .models import GCNdesign
from .dataset import pdb2input_jax, add_margin_jax
from .pdbutil import ProteinBackbone # Assumed to be framework-agnostic

# --- Constandts (unchanged) ---
i2aa = ('A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
        'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y')
aa2i = {aa: i for i, aa in enumerate(i2aa)}

# --- Helper Function (unchanged, uses NumPy) ---
def eliminate_restype(prob, unused_aas):
    """Masks and renormalizes probabilities to exclude certain amino acids."""
    mask = np.ones_like(prob, dtype=bool)
    for aa in unused_aas:
        if aa in aa2i:
            mask[:, aa2i[aa]] = False
    
    prob = prob * mask
    # Add a small epsilon to prevent division by zero if all probs in a row are masked
    sum_prob = prob.sum(axis=-1, keepdims=True) + 1e-9
    prob = prob / sum_prob
    return prob

# --- Ported Predictor Class ---

#@jax.jit(static_argnames=["train"])
def run_model(params, node_p, edge_flat, adjmat_p, train):
    return GCNdesign(hypara=HyperParam()).apply(params, node_p, edge_flat, adjmat_p, train=train, mutable=["batch_stats"])
    #return GCNdesign(hypara=HyperParam()).apply(params, node_p, edge_flat, adjmat_p, train=train)
run_model = jax.jit(run_model, static_argnames=["train"])

class PredictorJax():
    def __init__(self, param: str = None, hypara: HyperParam = None, load_state: bool = True, save_state: bool = False):
        """
        Initializes the JAX Predictor.

        Args:
            param: Path to the converted JAX parameters (.msgpack file).
            hypara: A HyperParam object.
        """
        self.hypara = hypara if hypara else HyperParam()
        self.param = param if param else './gcndesign_jax/params/param_default_jax.msgpack'
        # --- Model and Parameter Loading ---
        assert path.isfile(self.param), f"JAX parameter file not found: {self.param}"
        
        # 1. Instantiate the model       
        if load_state:
            self.model = None
            param_path = "gcndesign_jax/params/param_default_jax_variable.msgpack"
            with open(param_path, "rb") as f:
                self.variables = jax.device_put(from_bytes(None, f.read()))  # ← NO scaffold needed
        else:
            self.model = GCNdesign(hypara=self.hypara)
            N = 62  # arbitrary small number of nodes
            node_dim = 6
            edge_dim = 36
        
            # Dummy inputs matching original shapes/dtypes
            node = jnp.ones((N, node_dim), dtype=jnp.float32)
            edgemat = jnp.ones((N, N, edge_dim), dtype=jnp.float32)
            adjmat1 = jnp.zeros((N, N-20, 1), dtype=jnp.bool_)
            adjmat2 = jnp.ones((N, 20, 1), dtype=jnp.bool_)
            adjmat = jnp.concatenate([adjmat1, adjmat2], axis=1)
            label = jnp.ones((N,), dtype=jnp.int32)
            mask = jnp.ones((N,), dtype=jnp.bool_)
    
            node_p, edgemat_p, adjmat_p, _, _ = add_margin_jax(
                node, edgemat, adjmat, label, mask, self.hypara.nneighbor
            )
            key = jax.random.PRNGKey(0)
            variables_scaffold = self.model.init(
                {'params': key, 'dropout': key}, node_p, edgemat_p, adjmat_p, train=False
            )

            # 3. Load the actual parameters from file into the scaffold
            with open(self.param, 'rb') as f:
                self.variables = serialization.from_bytes(variables_scaffold, f.read())
                jax.debug.print("✅ JAX model and parameters loaded successfully.")
        if save_state:
            with open("model_variables.msgpack", "wb") as f:
                f.write(to_bytes(self.variables))

    def _pred_base(self, pdb: str):
        """Core prediction logic for a single PDB file."""
        # Input data setup using JAX-ported functions
        node, edgemat, adjmat, label, mask, aa1 = pdb2input_jax(pdb, self.hypara)
        node_p, edgemat_p, adjmat_p, _, _ = add_margin_jax(
            node, edgemat, adjmat, label, mask, self.hypara.nneighbor
        )
        
        adj = adjmat_p.squeeze(-1)  # (L, L) boolean                                                                                                       
        edge_flat = edgemat_p[adj]
    
        # Prediction using model.apply with loaded variables
        # The [1:-1] slice removes the margins, same as the original code
        if self.model is not None:
            outputs, latent = self.model.apply(self.variables, node_p, edge_flat, adjmat_p, train=False)        
        else:
            #((outputs, latent), batch_stats)  = run_model(self.variables, node_p, edge_flat, adjmat_p, False)
            ((outputs, latent), _)  = run_model(self.variables, node_p, edge_flat, adjmat_p, False)
            outputs.block_until_ready()
            latent.block_until_ready()
            jax.debug.print("run_model done")
        return outputs, aa1, latent

    def predict_logit_tensor(self, pdb: str, as_dict: bool = False):
        """Predicts raw logits for each residue."""
        assert path.isfile(pdb), f"PDB file not found: {pdb}"
        
        logit, _, latent  = self._pred_base(pdb)
        # Convert JAX array to a standard NumPy array for CPU-based processing
        logit_np = np.array(logit)
        if as_dict:
            return [dict(zip(i2aa, l)) for l in logit_np]
        return logit_np
    
    def predict(self, pdb: str, temperature: float = 1.0):
        """Predicts amino acid probabilities for each residue."""
        assert path.isfile(pdb), f"PDB file not found: {pdb}"
        
        pbb = ProteinBackbone(file=pdb)
        id2org = [(int(v[1:]), v[0]) for v in pbb.iaa2org]
        
        logit, aa1, latent = self._pred_base(pdb)
        jax.debug.print("executed")
        logit = logit[1:-1]
        latent = latent[1:-1]
        #latent = latent[1:-1]
        # Convert logits to probabilities using JAX softmax
        prob = jax.nn.softmax(logit / temperature, axis=1)

        prob_np = np.array(prob) # Convert to NumPy array
        jax.debug.print("prob got")
        pdict = [dict(zip(i2aa, p)) for p in prob_np]
        return [(p, {'resnum': v[0], 'chain': v[1], 'original': a}) for p, v, a in zip(pdict, id2org, aa1)]

    def make_resfile(self, pdb: str, temperature: float = 1.0, prob_cut: float = 0.8, unused=None):
        """Generates a Rosetta-compatible resfile."""
        assert path.isfile(pdb), f"PDB file not found: {pdb}"
        
        pbb = ProteinBackbone(file=pdb)
        id2org = [(int(v[1:]), v[0]) for v in pbb.iaa2org]
        
        logit, aa1 = self._pred_base(pdb)
        prob = jax.nn.softmax(logit / temperature, axis=1)
        prob_np = np.array(prob)
        
        if unused:
            prob_np = eliminate_restype(prob_np, unused)
            
        prob_with_ids = [(p, *v) for p, v in zip(prob_np, id2org)]
        
        # The rest of the logic is string formatting and remains the same
        line_resfile = 'start\n'
        for idx, (p, i, a) in enumerate(prob_with_ids):
            line_resfile += f' {i:4d} {a:1s} PIKAA  '
            pikaa = ''
            psum = 0.0
            sorted_args = np.argsort(-p)
            for j in range(20):
                iarg = sorted_args[j]
                pikaa += i2aa[iarg]
                psum += p[iarg]
                if j > 0 and psum > prob_cut:
                    break
            line_resfile += f'{pikaa:20s} # {aa1[idx]}\n'
            
        return line_resfile

    
