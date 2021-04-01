# `psycho_ai`: analyzing biases in AI models using Psychophysics

# Authors

- Lizhen Liang, Ph.D. student, Syracuse University
- Daniel E. Acuna, iSchool, Syracuse University

# Introduction

# Examples

Extract PSE and JND of word embedding models:

```python
from psycho_ai import two_afc
pse = two_afc.pse(target_words, left_words, right_words, embedding)
```

# Citation

Please refer to

- Liang, L., & Acuna, D. E. (2020, January)
  . [Artificial mental phenomena: Psychophysics as a framework to detect perception biases in AI models](https://dl.acm.org/doi/abs/10.1145/3351095.3375623)
  . In Proceedings of the 2020 Conference on Fairness, Accountability, and Transparency (
  pp. 403-412).

