import torch
from itertools import permutations


def sdr(x, s, eps=1e-8):
    """
    Calculate SDR (Signal-to-Distortion Ratio)
    Input:
        x: separated signal, N x S tensor
        s: reference signal, N x S tensor
        eps: small value to prevent division by zero
    Return:
        sdr: N tensor
    """
    if x.shape != s.shape:
        raise RuntimeError(
            f"Dimension mismatch when calculating SDR: {x.shape} vs {s.shape}")

    # Centering the signals
    x_zm = x - torch.mean(x, dim=-1, keepdim=True)
    s_zm = s - torch.mean(s, dim=-1, keepdim=True)

    # Signal power and error power
    signal_power = torch.sum(s_zm ** 2, dim=-1)
    error_power = torch.sum((x_zm - s_zm) ** 2, dim=-1)

    # SDR calculation
    sdr_value = 10 * torch.log10(signal_power / (error_power + eps))
    return sdr_value

def SDRLoss(ests, egs):
    """
    Calculate SDR-based loss
    Input:
        ests: estimated signals, spks x N x S tensor
        egs: ground truth signals, spks x N x S tensor
    Return:
        loss: scalar loss value
    """
    refs = egs
    num_spks = len(refs)

    def sdr_loss(permute):
        # For one permutation
        return sum([sdr(ests[s], refs[t]) for s, t in enumerate(permute)]) / len(permute)

    # Compute SDR for all permutations
    N = egs[0].size(0)
    sdr_mat = torch.stack(
        [sdr_loss(p) for p in permutations(range(num_spks))]
    )
    max_perutt, _ = torch.max(sdr_mat, dim=0)
    
    # Return negative SDR as loss
    return -torch.sum(max_perutt) / N