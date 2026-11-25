# PyTorch is the only array backend

Specflow is written against PyTorch alone. The JAX rewrite begun on the `refactor` and `update-physics-quantities` branches is not continued; its useful ideas (a frozen mesh, axes counted from the end, derivatives as mesh methods) are ported to PyTorch. Every earlier generation and the slab-decomposition prototype built on `torch.distributed` already run on PyTorch.
