# Data attribution

Demo assets in this directory use the **GTEA** (Google/Texas Ego-Centric
Action) dataset, subject S1, video `S1_Cheese_C1`.

Required citation:

> Yin Li, Miao Liu, and James M. Rehg. "In the Eye of the Beholder: Joint
> Learning of Gaze and Actions in First Person Video." ECCV, 2018.

- The egocentric footage and action labels are the original dataset data.
- The bounding boxes and localized crops derive from the GTEA **hand-mask
  annotations** (sparse XML polygon annotations, one per 40 original frames).
  Boxes between annotation frames use the conservative interpolation policy
  documented in `cooking-demo.json`.
- No model predictions are shown; all on-screen labels are dataset
  annotations.
