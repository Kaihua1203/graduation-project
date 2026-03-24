# Methods and Backbones (Minimal)

## Method Selection Entry

The `METHODS` dictionary in `solo/methods/__init__.py` maps the `method` config string to the implementation class.

Common methods available in this repository:

- simclr, simsiam
- byol, mocov2plus, mocov3
- vicreg, vibcreg, barlow_twins
- swav, deepclusterv2, dino, mae
- nnclr, nnsiam, nnbyol, ressl, supcon, wmse, all4one

## Backbone Entry

`_BACKBONES` in `solo/methods/base.py` provides selectable backbones:

- resnet18, resnet50
- vit_tiny/small/base/large
- swin_tiny/small/base/large
- convnext_tiny/small/base/large
- poolformer_s12/s24/s36/m36/m48
- wide_resnet28w2/wide_resnet28w8

## Practical Recommendation for This Project

- Start with one stable pair first (for example, `simclr + resnet18`).
- Change one axis at a time for ablation (fix method and swap backbone, or fix backbone and swap method).
