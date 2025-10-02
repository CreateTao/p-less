# *p*-less Sampling
This repository contains the code for the paper "*p*-less Sampling: A Robust Hyperparameter-Free Approach for LLM Decoding". The paper is available [here](https://arxiv.org/abs/2509.23234).

TL;DR: We introduce *p*-less Sampling: a hyperparameter-less and information-theoretic approach to sampling which dynamically sets a truncation threshold at each decoding step of the LLM, based on the entire token probability distribution. We further introduce *p*-less<sub>norm</sub>, a variant of *p*-less, which effectively relaxes the threshold and retains similar desirable properties as p-less, for tasks where diversity is favored over coherence.

## Examples
Refer to the [notebook](https://github.com/ryttry/p-less-sampling/blob/main/p_less_examples.ipynb) for working examples on *p*-less and *p*-less<sub>norm</sub> decoding.

> [!IMPORTANT]
> ***p-less and p-less<sub>norm</sub> samplers can be a direct drop-in to your LLM, see the [implementation](https://github.com/ryttry/p-less-sampling/blob/main/p_less_samplers.py) and [notebook](https://github.com/ryttry/p-less-sampling/blob/main/p_less_examples.ipynb) on how it is done!*** 🚀

#### Installation Requirements
`pip install torch transformers`

Tested with Python 3.10.12, torch 2.6.0 and transformers 4.55.2.

## Advantages of *p*-less (and *p*-less<sub>norm</sub>) over Existing LLM Decoding Methods
1. The truncation threshold utilized in *p*-less sampling dynamically adapts to the entire token probability distribution at each time step. In contrast, existing sampling methods either use a fixed threshold which ignores the current token probability distribution (e.g. top-p, top-k, ϵ-sampling), set the threshold based on the probability of a single token in the current distribution (e.g. min-p), or only considers the token distribution if conditions are met (e.g. η-sampling).
2. *p*-less produces a bounded and valid truncation threshold which guarantees a non-empty candidate set for sampling, unlike other sampling methods where bounds are not guaranteed and edge cases are resolved with defaults, such as defaulting to the modal token (or top few tokens) if all tokens do not meet the threshold (e.g. ϵ-sampling, η-sampling, mirostat).
3. The truncation threshold of p-less sampling dynamically adjusts with temperature, unlike other methods (e.g. top-p, top-k, min-p, ϵ-sampling) whose hyperparameters are not meaningful when temperature approaches zero or infinity.

Thus, *p*-less uniquely possesses all three of the aforementioned desirable properties of a sampling approach, combining the benefits of existing sampling strategies into a single method.

4. Generation efficiency: *p*-less is more efficient than other methods, both in terms of token sampling speed and overall generation length, without sacrificing task-specific performance.

We validated the effectiveness of *p*-less sampling through extensive experiments: using three LLMs and five datasets spanning math, logical reasoning, and creative writing tasks.

## Coming your way soon
We are working towards contributing the *p*-less samplers to common LLM inference APIs.

#### If you find our paper or code useful in your work, please cite it as:
```
@article{RunyanSP2025,
  title={p-LESS SAMPLING: A ROBUST HYPERPARAMETER-FREE APPROACH FOR LLM DECODING},
  author={Runyan Tan and Shuang Wu and Phillip Howard},
  journal={arXiv preprint arXiv:2509.23234},
  year={2025},
  note={[cs.AI, cs.CL]},
  url={https://arxiv.org/abs/2509.23234}
}
```
