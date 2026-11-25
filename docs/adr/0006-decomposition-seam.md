# Physics code goes through the decomposition

All FFTs, reductions over the spatial axes and gather/scatter for IO go through the mesh's decomposition; physics code never calls `torch.fft` or reduces over spatial axes itself. The first pass ships only the serial decomposition, but the slab prototype on `extend-to-distributed-architecture` had to patch 15 files because these calls were scattered. With one interface, a distributed decomposition can replace the serial one without touching the physics.
