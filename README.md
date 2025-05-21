This is a hard fork of https://github.com/allenai/satlas/tree/46e34afc8f241681f2d18a10e9f99a18e5d43df2

With adjustments made to the forward pass of the model so that it can be exported with AOTInductor. The output format of AOTInductor is now referred to commonly as ExportedProgram or .pt2. 


## Running Model export

This will create a uv environment with torch gpu dependencies (required to compile the model for NVIDIA hardware) and the satlas solar model dependencies which includes the hard fork in the src directory.

```
uv sync
```

To export the solar model, run this

```
uv run export.py test.pt2
```

this will save a file `test.pt2`

after some compile time you should get

```
/home/rave/work/satlas-aoti/.venv/lib/python3.11/site-packages/torch/fx/graph.py:1801: UserWarning: Node backbone_backbone_backbone_features_1_1_attn_lifted_tensor_4 target backbone.backbone.backbone.features.1.1.attn.lifted_tensor_4 lifted_tensor_4 of backbone.backbone.backbone.features.1.1.attn does not reference an nn.Module, nn.Parameter, or buffer, which is what 'get_attr' Nodes typically target
  warnings.warn(
/home/rave/work/satlas-aoti/.venv/lib/python3.11/site-packages/torch/fx/graph.py:1810: UserWarning: Additional 428 warnings suppressed about get_attr references
  warnings.warn(
AOT model compile time: 364.897384 seconds
AOT model inference time: 0.556527 seconds
```

This repo was set up to make it easier to reproduce and debug these issues
https://github.com/pytorch/pytorch/issues/153992
https://github.com/pytorch/pytorch/issues/146524

Two test files are included for each to reprouce the issues.
