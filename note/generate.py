"""Minimal script to load AMPDiffusion checkpoint and generate AMP sequences."""
import torch
import esm
import argparse

def generate(checkpoint_path, num_sequences=100, design_len=32, device='cuda'):
    """
    Generate AMP sequences from a trained AMPDiffusion checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint (.pt file)
        num_sequences: Number of sequences to generate
        design_len: Target sequence length (10-40 aa). Total tokens = design_len + 2 (cls + eos)
        device: 'cuda' or 'cpu'

    Returns:
        List of generated peptide sequences
    """
    import sys, os
    # Import from parent directory
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from ampdiffusion import Denoise_Transformer, GaussianDiffusion1D
    from ema_pytorch import EMA

    # Load ESM2-8M
    esm2, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    esm2.eval().to(device)

    # Build model (16.54M params total: 7.40M ESM2 layers + 9.14M own)
    model = Denoise_Transformer(esm_layers=esm2.layers, embed_dim=320, pep_max_len=42)
    diffusion = GaussianDiffusion1D(
        model, seq_length=42, timesteps=1000, objective='pred_x0',
        loss_type='l2', auto_normalize=False, embed_dim=320,
        self_condition=False, device=device,
    ).to(device)

    # Load checkpoint (EMA weights)
    data = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ema = EMA(diffusion, beta=0.999, update_every=10)
    ema.to(device)
    ema.load_state_dict(data['ema'])
    ema.ema_model.eval()

    # Decode setup: only use 20 standard amino acids
    standard_aa = list('ACDEFGHIKLMNPQRSTVWY')
    std_idxs = [alphabet.get_idx(aa) for aa in standard_aa]

    # Generate embeddings via diffusion sampling
    new = ema.ema_model.sample(batch_size=num_sequences, design_len=design_len + 2)

    # Decode embeddings to sequences via ESM2 lm_head
    sequences = []
    for i in range(num_sequences):
        with torch.no_grad():
            logits = esm2.lm_head(new[i, :])
            # Select only standard AA columns, argmax over those
            aa_logits = logits[:, std_idxs]
        pred = torch.argmax(aa_logits, dim=1)
        s = ''.join(standard_aa[j] for j in pred)
        s = s[:design_len]
        sequences.append(s)

    return sequences


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate AMPs with AMPDiffusion')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to .pt checkpoint')
    parser.add_argument('--num', type=int, default=100, help='Number of sequences')
    parser.add_argument('--length', type=int, default=30, help='Target peptide length (aa)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output', type=str, default='generated.csv')
    args = parser.parse_args()

    print(f"Generating {args.num} sequences of length {args.length}...")
    seqs = generate(args.checkpoint, args.num, args.length, args.device)

    import pandas as pd
    df = pd.DataFrame({'sequence': seqs})
    df.to_csv(args.output, index=False)
    print(f"Saved {len(seqs)} sequences to {args.output}")
