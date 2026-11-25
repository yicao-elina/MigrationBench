# Model provenance｜最终模型审计

## Final-model gate

三个 seed 都完成 1,000 个 requested epochs，并且各自只有一个 epoch-999 checkpoint。checkpoint 与导出的 `.model` 均有 90 个 state-dict tensor key，逐 tensor 最大绝对差值为 `0.0`；三个 compiled model 也存在。

| seed | training job | epoch | model SHA256 | exact match |
|---:|---|---:|---|---|
| 123 | `31584299` | 999 | `f4e280e7d7fd36921f84a911575245911f74dccf354c94e13cb3c9bf62101d7c` | yes |
| 456 | `31585322_456` | 999 | `a4193dc8049fa3f19d187cb6b8332710a992e1cc4d755568d27d6f48215ccc3c` | yes |
| 789 | `31585322_789` | 999 | `ad6a4d416d051619c68f3be7f148725af421337703e6752643e47f44b91e659f` | yes |

Foundation SHA256：`e618ad582b84239905b9c3b77ce6e9ce111b0ecd1533223a1a6aac7a696b8aa0`。

## Validation environment

- Python `3.9.22`
- PyTorch `2.7.0+cu118`
- MACE `0.3.12`
- device: A100 CUDA
- probed elements: Cr, Cu, Li, Na, Sb, Te

四个模型均加载成功，并对上述六种元素返回 finite energy/force probe。完整 provenance 位于 `results/provenance/`，总 manifest 位于 `manifests/model_manifest.json`。

模型二进制没有进入 Git commit；见 [[reproduce]] 和 `models/README.md`。
