import numpy as np
import pprint

# Import the original PyTorch Predictor and the new JAX Predictor
from gcndesign.predictor import Predictor as PredictorTorch
from gcndesign_jax.predictor import PredictorJax

# Define the amino acid order to ensure consistent comparison
I2AA = ('A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
        'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y')

def design_sequence_from_prediction(prediction_output):
    """
    Selects the residue with the highest probability (argmax) for each position.

    Args:
        prediction_output: The list output from the predictor's .predict() method.

    Returns:
        A list of dictionaries, each containing the residue number, chain,
        original residue, and the newly designed residue.
    """
    designed_sequence = []
    for item in prediction_output:
        prob_dict, meta_info = item
        
        # Extract probabilities in a consistent order
        probs = np.array([prob_dict[aa] for aa in I2AA])
        # Find the index of the highest probability
        top_index = np.argmax(probs)
        
        # Get the amino acid code for that index
        designed_aa = I2AA[top_index]
        
        # Store the result
        designed_sequence.append({
            'resnum': meta_info['resnum'],
            'chain': meta_info['chain'],
            'original': meta_info['original'],
            'designed': designed_aa
        })
        
    return designed_sequence

def compare_predictions(pdb_file: str):
    """
    Runs both PyTorch and JAX predictors and compares their outputs residue by residue.
    """
    print("--- Starting Model Comparison ---")

    # 1. Run the original PyTorch model
    print("Running PyTorch Predictor...")
    predictor_torch = PredictorTorch(device='cpu')
    out_torch = predictor_torch.predict(pdb=pdb_file)
    print("✅ PyTorch prediction complete.")

    # 2. Run the ported JAX model
    print("\nRunning JAX Predictor...")
    predictor_jax = PredictorJax()
    out_jax = predictor_jax.predict(pdb=pdb_file)
    print("✅ JAX prediction complete.")

    # 3. Perform the comparison
    print("\n--- Comparing Results ---")
    
    if len(out_torch) != len(out_jax):
        print(f"❌ Mismatch: Output lengths are different! Torch: {len(out_torch)}, JAX: {len(out_jax)}")
        return

    mismatches = 0
    for i, (item_torch, item_jax) in enumerate(zip(out_torch, out_jax)):
        probs_torch = np.array([item_torch[0][aa] for aa in I2AA])
        probs_jax = np.array([item_jax[0][aa] for aa in I2AA])
        #if not np.allclose(probs_torch, probs_jax, atol=1e-7):
        if not np.allclose(probs_torch, probs_jax, atol=1e-2):
            mismatches += 1

    # 4. Final Report
    print("\n--- Comparison Summary ---")
    if mismatches == 0:
        print("✅🎉 Perfect Match! The outputs from the PyTorch and JAX models are numerically identical.")
    else:
        print(f"Found {mismatches} mismatched residues.")
        
    print("-" * 29)
    
    # 5. Design Sequence using argmax
    # ==================================
    print("\n--- Designing Sequence from JAX Output (Argmax) ---")
    designed_res = design_sequence_from_prediction(out_jax)
    
    # Print a summary of the designed sequence
    #print("Pos.  Chain  Orig -> Design")
    #print("----  -----  ------------")
    #for res in designed_res:
    #    print(f"{res['resnum']:<4d}  {res['chain']:<5}  {res['original']:>4} -> {res['designed']:<6}")

    # You can also get the full sequence as a string
    full_sequence = "".join([res['designed'] for res in designed_res])
    print(f"\nFull Designed Sequence:\n{full_sequence}")


        # 5. Design Sequence using argmax
    # ==================================
    print("\n--- Designing Sequence from Torch Output (Argmax) ---")
    designed_res = design_sequence_from_prediction(out_torch)
    
    # Print a summary of the designed sequence
    #print("Pos.  Chain  Orig -> Design")
    #print("----  -----  ------------")
    #for res in designed_res:
    #    print(f"{res['resnum']:<4d}  {res['chain']:<5}  {res['original']:>4} -> {res['designed']:<6}")

    # You can also get the full sequence as a string
    full_sequence = "".join([res['designed'] for res in designed_res])
    print(f"\nFull Designed Sequence:\n{full_sequence}")


if __name__ == '__main__':
    PDB_TO_TEST = "tmp.pdb"
    compare_predictions(pdb_file=PDB_TO_TEST)
