import torch


def p_less_decode(
    probs: torch.Tensor,
) -> torch.Tensor:
    """
    Perform p-less sampling on a token probability distribution. Takes in a probability distribution over the
    vocabulary and returns the sampled token index.

    Args:
        probs (torch.Tensor): Probability distribution over the vocabulary, shape: (batch_size, vocabulary_size)

    Returns:
        torch.Tensor: Sampled token index, shape: (batch_size, 1)

    Note:
        p-less sampling admits tokens whose probability mass is at least the p-less threshold. The p-less
        threshold is hyperparameter-free and derived from the full distribution, therefore no hyperparameter
        tuning is required. The resulting distribution is then re-normalized based on the admitted tokens, after
        which sampling is performed. p-less is bounded and valid, i.e. at least one token will satisfy this
        constraint, therefore there is no need to defend against edge cases that could occur in other decoding
        methods. See https://arxiv.org/abs/2509.23234.
    """
    p = probs.square().sum(dim=-1, keepdim=True)
    mask = probs < p
    probs[mask] = 0.0
    probs.div_(probs.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token


def p_less_norm_decode(
    probs: torch.Tensor,
) -> torch.Tensor:
    """
    Perform p-less-norm sampling on a token probability distribution. Takes in a probability distribution over
    the vocabulary and returns the sampled token index.

    Args:
        probs (torch.Tensor): Probability distribution over the vocabulary, shape: (batch_size, vocabulary_size)

    Returns:
        torch.Tensor: Sampled token index, shape: (batch_size, 1)

    Note:
        p-less-norm sampling admits tokens whose probability mass is at least the p-less-norm threshold. The
        p-less-norm threshold is hyperparameter-free and derived from the full distribution, therefore no
        hyperparameter tuning is required. The resulting distribution is then re-normalized based on the
        admitted tokens, after which sampling is performed. p-less-norm is bounded and valid, i.e. at least one
        token will satisfy this constraint, therefore there is no need to defend against edge cases that could
        occur in other decoding methods. p-less-norm is relaxed from p-less and can be used for tasks where
        diversity is favored over coherence. See https://arxiv.org/abs/2509.23234.
    """
    v = probs.size(-1)
    p = (v * probs.square().sum(dim=-1, keepdim=True) - 1.0) / (v - 1.0)
    mask = probs < p
    probs[mask] = 0.0
    probs.div_(probs.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token
