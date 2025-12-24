import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


@torch.no_grad()
def _score2mask_nm_kernel(
    mask_score: torch.Tensor,
    prune_M: int,
    prune_N: int,
) -> torch.Tensor:
    """
    Compute (N:M) group-wise pruning mask for a 2D array of scores.

    Args:
        mask_score: (prune_num_groups, prune_M)
        prune_M: group size (M)
        prune_N: number kept per group (N)

    Returns:
        mask: (prune_num_groups, prune_M)
    """
    # Keep top-N (i.e., zero out the smallest M-N per group)
    k = int(prune_M - prune_N)
    index = torch.argsort(mask_score, dim=1, descending=False)[:, :k]  # (groups, M-N)
    mask = torch.ones_like(mask_score)
    mask.scatter_(dim=1, index=index, value=0.0)
    return mask


@torch.no_grad()
def score2mask_nm(
    mask_score: torch.Tensor,
    prune_num_groups: int,
    prune_N: int,
    prune_M: int,
) -> torch.Tensor:
    """
    Transform 4D score tensor into grouped (N:M) mask, then flatten per filter.

    Args:
        mask_score: (Co, Ci, k, k)
        prune_num_groups: number of groups (= Co * k * k when grouping over Ci)
        prune_N: number kept per group
        prune_M: group size

    Returns:
        mask: (Co, Ci * k * k)
    """
    Co, Ci, k, _ = mask_score.size()

    # (Co, Ci, k, k) -> (Co, k, k, Ci) -> (prune_num_groups, prune_M)
    score_g = mask_score.permute(0, 2, 3, 1).reshape(prune_num_groups, prune_M)

    mask_g = _score2mask_nm_kernel(score_g, prune_M, prune_N)

    # Back to (Co, Ci, k, k) then flatten to (Co, Ci * k * k)
    mask = mask_g.reshape(Co, k, k, Ci).permute(0, 3, 1, 2).contiguous()
    return mask.reshape(Co, -1)


@torch.no_grad()
def score2mask_unstructured(mask_score: torch.Tensor, target_density: float) -> torch.Tensor:
    """
    Unstructured pruning mask from global per-filter threshold.

    Args:
        mask_score: (Co, Ci, k, k)
        target_density: fraction of weights to keep (0..1)

    Returns:
        mask: (Co, Ci * k * k)
    """
    Co, _, _, _ = mask_score.size()

    # Keep the top 'target_density' fraction (per entire tensor)
    thr = torch.quantile(mask_score, 1.0 - target_density, interpolation="linear")
    mask = mask_score.ge(thr).float()
    return mask.reshape(Co, -1)


def _quantize_symmetric(x: torch.Tensor, alpha: torch.Tensor, bits: int) -> torch.Tensor:
    """
    Symmetric linear quantization to [-alpha, alpha], with 2^bits levels.
    For activations bounded on [0, alpha], caller should clamp before passing.
    """
    # scaling = (2^bits - 1) / alpha  (same as your code)
    scaling = (2 ** bits - 1) / alpha
    return torch.round(torch.clamp(x, min=-alpha, max=alpha) * scaling) / scaling


def _quantize_activation(x: torch.Tensor, alpha: torch.Tensor, bits: int) -> torch.Tensor:
    """
    Activation quantization used in your code: clamp to [0, alpha] then quantize.
    """
    scaling = (2 ** bits - 1) / alpha
    return torch.round(torch.clamp(x, max=alpha) * scaling) / scaling


def adder2d_function(
    X: torch.Tensor,
    W: torch.Tensor,
    alpha_quant: torch.Tensor,
    bits: int,
    stride: int,
    padding: int,
    mask_score: torch.Tensor,
    mask: torch.Tensor,
    use_transform: int,
    update_score: int,
) -> torch.Tensor:
    """
    2D adder-conv wrapper that unfolds input and calls the custom Function.
    Shapes:
      X: (N, Cin, H, W)
      W: (Cout, Cin, Kh, Kw)
    """
    n_filters, _, h_filter, w_filter = W.size()
    n_x, _, h_x, w_x = X.size()

    h_out = int((h_x - h_filter + 2 * padding) / stride + 1)
    w_out = int((w_x - w_filter + 2 * padding) / stride + 1)

    # (N, Cin*Kh*Kw, H_out*W_out) -> (Cin*Kh*Kw, H_out*W_out*N)
    X_col = F.unfold(X, (h_filter, w_filter), dilation=1, padding=padding, stride=stride)
    X_col = X_col.permute(1, 2, 0).contiguous().view(X_col.size(1), -1)

    # (Cout, Cin*Kh*Kw)
    W_col = W.view(n_filters, -1)

    # (Cout, Cin*Kh*Kw)
    mask_score_reshape = mask_score.view(n_filters, -1)

    # Forward through custom Function -> (Cout, H_out*W_out*N)
    out = _Adder.apply(
        W_col, X_col, alpha_quant, bits, mask, mask_score_reshape, use_transform, update_score
    )

    # Reshape back to (N, Cout, H_out, W_out)
    out = out.view(n_filters, h_out, w_out, n_x).permute(3, 0, 1, 2).contiguous()
    return out


class _Adder(Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        W_col: torch.Tensor,
        X_col: torch.Tensor,
        alpha_quant: torch.Tensor,
        bits: int,
        mask: torch.Tensor,
        mask_score_reshape: torch.Tensor,  # kept for saved-tensors compatibility
        use_transform: int,
        update_score: int,
    ) -> torch.Tensor:
        """
        W_col: (Cout, D)
        X_col: (D, S)              where S = steps * batch
        mask:  (Cout, D)           binary mask
        """
        ctx.update_score = update_score

        if use_transform == 1:
            if bits == 0:
                raise NotImplementedError("use_transform = 1 should not be used with bits=0")

            # Quantize as in your original branches
            X_col_q = _quantize_activation(X_col, alpha_quant, bits)
            W_col_q = _quantize_symmetric(W_col, alpha_quant, bits)

            W_col_q_pos = torch.clamp(W_col_q, min=0.0)

            # post_term: (Cout, S)
            post_term_intermediate = X_col_q.unsqueeze(0) - 2.0 * torch.min(
                W_col_q_pos.unsqueeze(2), X_col_q.unsqueeze(0)
            )
            post_term = (post_term_intermediate * mask.unsqueeze(2)).sum(1)

            # pre_term: (Cout, 1)
            pre_term = torch.sum(torch.abs(W_col) * mask, dim=1, keepdim=True)

            output = -(pre_term + post_term)  # (Cout, S)
            ctx.save_for_backward(W_col, X_col, mask)
            return output

        # Non-transform path
        if bits == 0:
            diff = (W_col.unsqueeze(2) - X_col.unsqueeze(0)).abs()  # (Cout, D, S)
        else:
            X_col_q = _quantize_activation(X_col, alpha_quant, bits)
            W_col_q = _quantize_symmetric(W_col, alpha_quant, bits)
            diff = (W_col_q.unsqueeze(2) - X_col_q.unsqueeze(0)).abs()

        diff = diff * mask.unsqueeze(2)
        output = -(diff.sum(1))  # (Cout, S)
        ctx.save_for_backward(W_col, X_col, mask)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        grad_output: (Cout, S)
        Returns grads for:
          W_col, X_col, alpha_quant(None), bits(None),
          mask(None), mask_score_reshape(or grad), use_transform(None), update_score(None)
        """
        W_col, X_col, mask = ctx.saved_tensors
        update_score = ctx.update_score

        # Shared diff: shape (Cout, D, S)
        diff = X_col.unsqueeze(0) - W_col.unsqueeze(2)

        if update_score == 1:
            diff_mask = diff * mask.unsqueeze(2)
            # grad wrt X: (-diff_mask) clamped, contract over Cout
            grad_X_col = ((-diff_mask).clamp(-1, 1) * grad_output.unsqueeze(1)).sum(0)
            # grad wrt W: contract over S
            grad_W_col = (diff_mask * grad_output.unsqueeze(1)).sum(2)
            # normalize as in your code
            grad_W_col = (
                grad_W_col
                / grad_W_col.norm(p=2).clamp(min=1e-12)
                * math.sqrt(W_col.size(1) * W_col.size(0))
                / 5
            )
            # importance for mask scores (mean over S)
            grad_mask_score_reshape = diff.abs().mean(2)
        else:
            grad_X_col = ((-diff).clamp(-1, 1) * grad_output.unsqueeze(1)).sum(0)
            grad_W_col = (diff * grad_output.unsqueeze(1)).sum(2)
            grad_W_col = (
                grad_W_col
                / grad_W_col.norm(p=2).clamp(min=1e-12)
                * math.sqrt(W_col.size(1) * W_col.size(0))
                / 5
            )
            grad_mask_score_reshape = None

        return (
            grad_W_col,
            grad_X_col,
            None,  # alpha_quant
            None,  # bits
            None,  # mask
            grad_mask_score_reshape,  # mask_score_reshape
            None,  # use_transform
            None,  # update_score
        )


class adder2d_qp(nn.Module):
    """
    Quantized + Prunable adder-2d layer.

    Args:
      input_channel, output_channel: ints
      args: expects attributes
        - quant_alpha (float)
        - quant_bits (int)
        - prune_M (int)
        - prune_unstructured (int, {0,1})
        - use_transform (int, {0,1})
        - update_score (int, {0,1})
    """

    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        args,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
    ):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.kernel_size = kernel_size

        # weights
        self.adder = nn.Parameter(
            nn.init.normal_(torch.randn(output_channel, input_channel, kernel_size, kernel_size))
        )

        # bias (fix: avoid truth-value testing of tensors in forward)
        self.bias = nn.Parameter(torch.zeros(output_channel)) if bias else None

        # quantization
        self.alpha = nn.Parameter(data=torch.tensor(args.quant_alpha), requires_grad=False)
        self.bits = args.quant_bits

        # pruning (scores are learnable; mask is a buffer)
        self.score_nm_mask = nn.Parameter(torch.zeros_like(self.adder), requires_grad=True)
        self.register_buffer(
            "prune_nm_mask",
            torch.ones(
                size=(self.adder.size(0), int(self.adder.size(1) * self.adder.size(2) * self.adder.size(3))),
                requires_grad=False,
            ),
        )
        self.prune_N = None
        self.prune_M = args.prune_M
        self.prune_num_groups = int(self.adder.numel() / args.prune_M)  # #groups with size M
        self.prune_unstructured = args.prune_unstructured
        self.prune_apply = False

        # adder function flags
        self.use_transform = args.use_transform
        self.update_score = args.update_score

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.set_mask()
        out = adder2d_function(
            x,
            self.adder,
            self.alpha,
            self.bits,
            self.stride,
            self.padding,
            self.score_nm_mask,
            self.prune_nm_mask,
            self.use_transform,
            self.update_score,
        )
        if self.bias is not None:
            # (Cout,) -> (1, Cout, 1, 1)
            out = out + self.bias.view(1, -1, 1, 1)
        return out

    @torch.no_grad()
    def set_mask(self):
        """
        Compute and cache the binary pruning mask in self.prune_nm_mask.
        """
        if self.prune_apply:
            if self.prune_unstructured == 1:
                mask = score2mask_unstructured(self.score_nm_mask, self.prune_N / self.prune_M)
            else:
                mask = score2mask_nm(
                    self.score_nm_mask, self.prune_num_groups, self.prune_N, self.prune_M)
            self.prune_nm_mask.data = mask
        else:
            self.prune_nm_mask.data = torch.ones_like(self.prune_nm_mask)

    @torch.no_grad()
    def reset_score_nm_mask(self):
        """
        Zero-out pruning scores (useful when restarting pruning).
        """
        self.score_nm_mask.data = torch.zeros_like(self.score_nm_mask)

    def __repr__(self) -> str:
        return (
            f"adder conv2d: Cin {self.input_channel} "
            f"Cout {self.output_channel} kernel {self.kernel_size} stride: {self.stride}"
        )
