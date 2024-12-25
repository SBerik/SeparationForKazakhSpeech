import torch
from itertools import permutations

def sisnr(x, s, eps=1e-8):
    """
    calculate training loss
    input:
          x: separated signal, N x S tensor
          s: reference signal, N x S tensor
    Return:
          sisnr: N tensor
    """

    def l2norm(mat, keepdim=False):
        return torch.norm(mat, dim=-1, keepdim=keepdim)

    if x.shape != s.shape:
        raise RuntimeError(
            "Dimention mismatch when calculate si-snr, {} vs {}".format(
                x.shape, s.shape))
    x_zm = x - torch.mean(x, dim=-1, keepdim=True)
    s_zm = s - torch.mean(s, dim=-1, keepdim=True)
    t = torch.sum(
        x_zm * s_zm, dim=-1,
        keepdim=True) * s_zm / (l2norm(s_zm, keepdim=True)**2 + eps)
    return 20 * torch.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps))


def SISNR_Loss(ests, egs):
    # spks x n x S
    refs = egs
    num_spks = len(refs)

    def sisnr_loss(permute):
        # for one permute
        return sum([sisnr(ests[s], refs[t]) for s, t in enumerate(permute)]) / len(permute)
             # average the value

    # P x N
    N = egs[0].size(0)
    sisnr_mat = torch.stack(
        [sisnr_loss(p) for p in permutations(range(num_spks))])
    max_perutt, _ = torch.max(sisnr_mat, dim=0)
    # si-snr
    return -torch.sum(max_perutt) / N


# % Formula for Scale-Invariant Signal-to-Noise Ratio (SI-SNR)

# \documentclass{article}
# \usepackage{amsmath}
# \usepackage{amssymb}

# \begin{document}

# \section*{Scale-Invariant Signal-to-Noise Ratio (SI-SNR)}

# The Scale-Invariant Signal-to-Noise Ratio (SI-SNR) is computed as:

# \[
# \text{SI-SNR}(x, s) = 20 \cdot \log_{10}\left(\epsilon + \frac{\|t\|_2}{\|x_{\text{zm}} - t\|_2 + \epsilon}\right)
# \]

# where:
# \begin{align*}
# x_{\text{zm}} &= x - \frac{1}{S} \sum x, \quad \text{(Zero-mean of separated signal)} \\
# s_{\text{zm}} &= s - \frac{1}{S} \sum s, \quad \text{(Zero-mean of reference signal)} \\
# t &= \frac{\sum(x_{\text{zm}} \cdot s_{\text{zm}})}{\|s_{\text{zm}}\|_2^2 + \epsilon} \cdot s_{\text{zm}}, \quad \text{(Projection of $x_{\text{zm}}$ onto $s_{\text{zm}}$)} \\
# \| \cdot \|_2 &= \text{L}_2\text{-norm}, \\
# \epsilon &= \text{A small constant to prevent division by zero.}
# \end{align*}

# \section*{Loss Function}

# For multiple speakers, the loss function is defined as the negative average of the maximum SI-SNR across all permutations of speaker assignments:

# \[
# \text{Loss} = -\frac{1}{N} \sum_{n=1}^N \max_{\pi \in \mathcal{P}} \frac{1}{|\pi|} \sum_{(s, t) \in \pi} \text{SI-SNR}(x_s, s_t)
# \]

# where:
# \begin{itemize}
#     \item $\mathcal{P}$: Set of all permutations of speaker indices.
#     \item $N$: Number of samples.
# \end{itemize}

# \end{document}
