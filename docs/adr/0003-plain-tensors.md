# Fields are plain tensors

Fields are plain `torch.Tensor`s, with no wrapper type carrying their rank or space. The `add-tensor-wrappers` branch tried `Field[Rank, Space]` generics: every operation had to re-wrap its result and every signature grew heavier, without the constraints being enforced in practice. Instead, a field's space is in its name (`velocity_spectral`, `vorticity_physical`), its layout follows one convention (spatial axes last, component axes just before them, leading axes free), and tight docstrings state what a name cannot.
