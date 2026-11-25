## Agent skills

- Issues: GitHub Issues (`ArtCouteau/specflow-solver`), labels `ready-for-agent` and `ready-for-human`. See `docs/agents/issue-tracker.md`.
- Domain: glossary in `GLOSSARY.md`, notation in `NOTATION.md`, decisions in `docs/adr/` (one short paragraph per ADR). Read them before writing specs or tickets, use the glossary's terms exactly, follow the conventions of the notation, and ask before contradicting an ADR.

## Conventions

- PEP 8 and ruff-clean. PascalCase for classes, snake_case for everything else.
- Explicit dependencies through constructors; no global context. Module loggers (`logging.getLogger(__name__)`) are the only accepted global; the library never configures logging.
- Plain `torch.Tensor`s. Spatial axes last, component axes just before them, leading axes free. Field names end with their space: `_spectral` or `_physical`. State keys are bare variable names.
- In-place operations wherever possible, on tensors the function allocated itself; never modify an argument, except an output tensor passed for that purpose (an `out_*` parameter).
- Google-style docstrings, kept tight, with math in inline TeX following the conventions of `NOTATION.md`; see `docs/adr/0007-tight-docstrings-with-inline-tex.md`. Python sources are ASCII only.
- No comments, no `# noqa`, no `# type: ignore`. If one seems necessary, stop and ask.
- Strict pyright on this code; typing gaps in torch or h5py are handled in configuration, never inline.
- N is even, float64 is the default precision, and every mesh array is in the mesh's dtype and device.
