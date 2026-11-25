# PyTorch is the only array backend

Specflow is written against PyTorch alone. The JAX rewrite begun on the `refactor` and `update-physics-quantities` branches is not continued; its useful ideas (a frozen mesh, axes counted from the end, derivatives as mesh methods) are ported to PyTorch. Every earlier generation (psns, 10_Specflow, the v1 code) and the slab-decomposition prototype built on `torch.distributed` already run on PyTorch.
