# FDDI Runtime Assets

The FDDI source repository is intentionally kept separate from large runtime
assets. Set `FDDI_ASSET_ROOT` to the external asset directory when it is not
located next to the repository.

Expected layout:

```text
FDDI_assets/
  models/modelscope/xichen/MimicBrush/
  models/modelscope/xichen/cleansd/
  datasets/VisA/
  datasets/KolektorSDD2/
  datasets/btad-DatasetNinja.tar
  results/mvtec/
  results/mvtec_new/
  results/VisA/
  cache/temp_fid_crops/
```

On the research server, this directory is `/home/zsf/FDDI_assets`. The old
paths under `/home/zsf/MimicBrush` are compatibility symlinks and remain valid
for the existing benchmark scripts.

The repository does not include model weights, full datasets, temporary cache
files, or bulk-generated images. Keep these assets out of GitHub and back them
up separately.
