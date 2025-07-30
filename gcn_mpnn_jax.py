import jax
import jax.numpy as jnp
import haiku as hk
import joblib
import os

from proteinmpnn_jax.modules import RunModel_Raw
from proteinmpnn_jax.colabdesign.af.prep import prep_pdb, prep_pos
from proteinmpnn_jax.colabdesign.af.alphafold.common import residue_constants
from proteinmpnn_jax.model import _aa_convert
from gcndesign_jax.predictor import PredictorJax as Predictor

# --- config ---
in_pdb_path = "model.pdb"
model_name = "v_48_020"
temperature_mpnn = 0.25
temperature_gcn = 0.25
noise = 0.1
rm_aa = "C"
#fixed_positions = [1, 2, 3, 4, 5]
fixed_positions = []
batch = 8

# --- setup model ---
model_path = os.path.join(os.path.dirname(__file__), "proteinmpnn_jax", "weights", f"{model_name}.pkl")
checkpoint = joblib.load(model_path)

config = {
    'num_letters': 21,
    'node_features': 128,
    'edge_features': 128,
    'hidden_dim': 128,
    'num_encoder_layers': 3,
    'num_decoder_layers': 3,
    'augment_eps': noise,
    'k_neighbors': checkpoint['num_edges'],
    'dropout': 0.0
}

model = RunModel_Raw(config)
params = jax.tree_util.tree_map(jnp.array, checkpoint["model_state_dict"])

# --- prepare inputs ---
pdb = prep_pdb(in_pdb_path)
atom_idx = tuple(residue_constants.atom_order[k] for k in ["N", "CA", "C", "O"])
chain_idx = jnp.concatenate([jnp.full((l,), i) for i, l in enumerate(pdb["lengths"])])
lengths = pdb["lengths"]
L = sum(lengths)

inputs = {
    "X": pdb["batch"]["all_atom_positions"][:, atom_idx],
    "mask": pdb["batch"]["all_atom_mask"][:, 1],
    "S": pdb["batch"]["aatype"],
    "residue_idx": pdb["residue_index"],
    "chain_idx": chain_idx,
    "lengths": jnp.array(lengths),
    "bias": jnp.zeros((L, 20)),
    "temperature": temperature_mpnn
}

# --- decoding order ---
randn = jax.random.uniform(jax.random.PRNGKey(0), (L,))
randn = jnp.where(inputs["mask"], randn, randn + 1)
inputs["decoding_order"] = randn.argsort()

# --- fix specific AAs ---
if fixed_positions:
    pos = prep_pos(",".join(map(str, fixed_positions)), **pdb["idx"])["pos"]
    inputs["fix_pos"] = pos
    fixed_aa = jax.nn.one_hot(inputs["S"], 21)[pos, :20]
    inputs["bias"] = inputs["bias"].at[pos].set(1e7 * fixed_aa)

# --- remove specific AAs ---
aa_order = residue_constants.restype_order
for aa in rm_aa.split(","):
    inputs["bias"] = inputs["bias"].at[:, aa_order[aa]].add(-1e6)

# --- convert to MPNN format ---
inputs["S"] = _aa_convert(inputs["S"])
inputs["bias"] = _aa_convert(inputs["bias"])

# --- run GCN predictor ---
for ii in range(2):
    if ii == 0:
        print("RAW MPNN")
    else:
        print("GCN MPNN")
        gcndes = Predictor()
        logit,aa1, latent  = gcndes._pred_base(pdb=in_pdb_path)
        logit = logit[1:-1]
        #prob = jax.nn.softmax(logit / temperature_gcn, axis=1)
        prob_padded = jnp.pad(logit, ((0, 0), (0, 1)))
        print(inputs["bias"].shape)
        print(prob_padded.shape)
        inputs["bias"] += 10*prob_padded # size mismatch
    # --- sample ---
    keys = jax.random.split(jax.random.PRNGKey(42), batch)
    sampled = jax.vmap(lambda k: model.sample(params, k, inputs))(keys)
    # --- convert to sequences ---
    seqs = []
    mpnn_alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    order_aa = {i: aa for i, aa in enumerate(mpnn_alphabet)}
    for S in sampled["S"]:
        seq = "".join([order_aa[int(jnp.argmax(s))] for s in S])
        seqs.append(seq)
    # --- output ---
    for i, seq in enumerate(seqs):
        print(f">sample_{i}\n{seq}")

    from pyfaspr import run_FASPR
    if ii == 0:
        nm = "RAW-MPNN"
    else:
        nm = "GCN-MPNN"
    for i in range(0, batch):
        pdb_text_out = run_FASPR(pdb=in_pdb_path, sequence=seqs[i])
        out_pdb_path = "3helix_bundle_GBB_design_" + nm + str(i) + ".pdb"
        with open(out_pdb_path, 'w') as f:
            f.write(pdb_text_out)

#def loss_fn(bias):
#    inputs_mod = {**inputs, "bias": bias}
#    out = model.sample(params, jax.random.PRNGKey(0), inputs_mod)
#    return -jnp.sum(out["logits"])  # dummy loss

# --- Compute gradients ---
#grad_fn = jax.grad(loss_fn)
#grads = grad_fn(prob_padded)

#print("Grad shape:", grads.shape)
#print("Grad sample:", grads[:5])
#print(type(prob_padded), type(prob_padded[0,0]))
#print(jax.dtypes.finfo(prob_padded.dtype))

