## Agent skills

- Issues: GitHub Issues (`ArtCouteau/specflow-solver`), labels `ready-for-agent` and `ready-for-human`. See `docs/agents/issue-tracker.md`.
- Domain: glossary in `GLOSSARY.md`, decisions in `docs/adr/` (one short paragraph per ADR). Read both before writing specs or tickets, use the glossary's terms exactly, and ask before contradicting an ADR.

## Conventions

- PEP 8 and ruff-clean. PascalCase for classes, snake_case for everything else.
- Explicit dependencies through constructors; no global context. Module loggers (`logging.getLogger(__name__)`) are the only accepted global; the library never configures logging.
- Plain `torch.Tensor`s. Spatial axes last, component axes just before them, leading axes free. Field names end with their space: `_spectral` or `_physical`. State keys are bare variable names.
- In-place operations wherever possible, on tensors the function allocated itself; never modify an argument.
- Google-style docstrings, kept tight: a one-line summary, and arguments or returns only for what names do not say (layout, space, units).
- No comments, no `# noqa`, no `# type: ignore`. If one seems necessary, stop and ask.
- Strict pyright on this code; typing gaps in torch or h5py are handled in configuration, never inline.
- N is even, float64 is the default precision, and every mesh array is in the mesh's dtype and device.
