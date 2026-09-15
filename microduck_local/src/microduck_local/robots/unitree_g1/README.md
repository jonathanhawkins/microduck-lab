# Unitree G1 (world person)

The follow scenes can attach this robot instead of the mocap capsule.

- **Body:** Lucky Robots `g1.xml` (29-DoF walk + fingers the policy ignores)
- **Brain:** their shipped `walker.onnx` (99-d obs → 29 joint targets, 50 Hz)
- **Meshes / ONNX:** not in git (~140 MB). `uv run fetch-g1` clones them into
  `microduck_local/.cache/unitree_g1/`.

`model_config.json` here is the joint order, default pose and action scales
the walker was trained with. Do not reorder it.
