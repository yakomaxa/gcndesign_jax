import numpy as np
import jax.numpy as jnp
from .pdbutil import ProteinBackbone as pdb
from .hypara import HyperParam
#from jax import device_put
import jax
# Int code of amino-acid types
mapped = {
    'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4,
    'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
    'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
    'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19
}
# 3 letter code to 1 letter code
three2one = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU':'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

def pdb2input_jax(filename: str, hypara: HyperParam):
    """
    Parse a PDB file into JAX-friendly inputs:
      - node:     (L, 6) float32 angles
      - edgemat:  (L, L, 36) float32 normalized distances
      - adjmat:   (L, nneighbor) int32 neighbor indices
      - label:    (L,) int32 amino acid codes
      - mask:     (L,) bool chain-break mask
      - aa1:      (L,) numpy.str_ one-letter sequence
    """
    # Load backbone and add missing atoms
    bb = pdb(file=filename)
    bb.addCB(force=True)
    bb.addH(force=True)
    bb.addO(force=True)
    bb.coord[0, 5] = bb.coord[0, 0]
    bb.coord[-1, 4] = bb.coord[-1, 3]

    L = len(bb)
    # Node features: dihedral angles (phi, psi, omega)
    node = np.zeros((L, 6), dtype=np.float32)
    bb.calc_dihedral()
    sins = np.sin(np.deg2rad(bb.dihedral))
    coss = np.cos(np.deg2rad(bb.dihedral))
    node[:, 0::2] = sins
    node[:, 1::2] = coss
    node[0, 0:2] = 0.0
    node[-1, 2:] = 0.0

    # Chain-break mask
    mask = np.ones((L,), dtype=bool)
    for i in range(L):
        d1 = np.linalg.norm(bb[i,0] - bb[i,1])
        d2 = np.linalg.norm(bb[i,1] - bb[i,2])
        if d1 > hypara.dist_chbreak or d2 > hypara.dist_chbreak:
            mask[i] = False
    for i in range(L-1):
        d3 = np.linalg.norm(bb[i,2] - bb[i+1,0])
        if d3 > hypara.dist_chbreak:
            mask[i] = mask[i+1] = False

    # Edge features & adjacency
    edgemat = np.zeros((L, L, 36), dtype=np.float32)
    nn_idx = bb.get_nearestN(hypara.nneighbor, atomtype='CB')  # (L, nneighbor)
    adjmat = np.zeros((L,L,1), dtype=bool)
    for i in range(L):
        adjmat[i,nn_idx[i]] = True
        if not mask[i]:
            continue
        for j in nn_idx[i]:
            # pairwise atom distances: (6,6) -> flatten 36
            dist = np.linalg.norm(
                bb[i,:,None,:] - bb[j,None,:,:],
                axis=2
            )
            edgemat[i, j] = (dist.flatten() - hypara.dist_mean) / hypara.dist_var

    # Labels
    res = bb.resname
    aa1 = np.array([three2one.get(x, 'X') for x in res], dtype='<U1')
    label = np.array([mapped.get(x, 20) for x in aa1], dtype=np.int32)
    mask = mask & (label != 20)

    # Convert to JAX arrays
    node     = jax.device_put(jnp.array(node))
    edgemat  = jax.device_put(jnp.array(edgemat))
    # adjacency as indices
    adjmat   = jax.device_put(jnp.array(adjmat))
    mask     = jax.device_put(jnp.array(mask))
    label    = jax.device_put(jnp.array(label))

    return node, edgemat, adjmat, label, mask, aa1

def add_margin_jax(node, edgemat, adjmat, label, mask, nneighbor):
    # Pad node: (0,0,1,1) means pad dim=0 with (1,1), dim=1 with (0,0)
    node = jnp.pad(node, ((1,1), (0,0)), mode='constant', constant_values=0)

    # Pad edgemat: shape (L, L, d_edge) → pad dim 0 & 1 with (1,1), dim 2 with (0,0)
    edgemat = jnp.pad(edgemat, ((1,1), (1,1), (0,0)), mode='constant', constant_values=0)

    # Pad adjmat: shape (L, L, 1) → pad dim 0 & 1 with (1,1), dim 2 with (0,0)
    adjmat = jnp.pad(adjmat, ((1,1), (1,1), (0,0)), mode='constant', constant_values=False)

    # Manually set head/tail padding region as True
    adjmat = adjmat.at[0, 1:nneighbor+1, 0].set(True)
    adjmat = adjmat.at[-1, 1:nneighbor+1, 0].set(True)

    # Pad label: shape (L,) → pad with value 20
    label = jnp.pad(label, (1,1), mode='constant', constant_values=20)

    # Pad mask: shape (L,) → pad with False
    mask = jnp.pad(mask, (1,1), mode='constant', constant_values=False)

    print("node", node.shape)
    print("edge", edgemat.shape)
    print("adj", adjmat.shape)

    return node, edgemat, adjmat, label, mask

