# License and provenance notice

- Original P2I source code: MIT, see LICENSE.
- `ref_physics_attention.py` and `upstream/Physics_Attention.py`: THUML MIT source;
  retain `experiments/transolver_lab/upstream/LICENSE` and its original copyright.
- PhysicsNeMo is an external dependency; its own license continues to apply.
- NVIDIA pretrained weights are downloaded directly from
  https://huggingface.co/nvidia/transolver_drivaerml and are not included here.
  Their model license is separate from this repository's source license.
- DrivAerML data: Neil Ashton et al., CC BY-SA 4.0,
  https://huggingface.co/datasets/neashton/drivaerml,
  https://arxiv.org/abs/2408.11969. Raw meshes/fields are not redistributed.
- Viewer screenshots adapt DrivAerML geometry and fields with P2I rendering,
  clipping, coloring and surrogate overlays. These screenshots and the technical
  report containing them are provided under CC BY-SA 4.0:
  https://creativecommons.org/licenses/by-sa/4.0/ .
- MIT does not relicense NVIDIA weights, external dependencies or dataset-derived
  figures. No institutional endorsement is implied.

Use of AI assistance: OpenAI Codex assisted implementation, debugging, experiment
orchestration and manuscript preparation. The named human author is responsible
for final review and publication decisions. Reported measurements come from
executed remote experiments, not model-generated estimates.
