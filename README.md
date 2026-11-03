# OptiSAR

Temporal land-cover segmentation that fuses Sentinel-1 and Sentinel-2 with a U-TAE. Each patch is 4 dates by 12 channels (10 optical bands plus VV and VH) at 256×256. The model maps every pixel to one of 6 land-cover classes. The test tile, 31UEQ, is held out of training.

The U-TAE is a spatial encoder shared across dates, a lightweight temporal attention block (L-TAE), and a U-Net decoder. Training has two stages: the decoder is trained with the encoder and L-TAE frozen, then every weight is fine-tuned from that checkpoint. The loss is weighted cross-entropy.

On the held-out tile the full model reaches weighted F1 **0.9387** and kappa **0.5757**. A ConvLSTM + Inception baseline on the same split scores 0.9037 and 0.4424.

Satellite imagery and ground-reference masks are the public MultiSenGE dataset and are not in this repository.

The temporal blocks follow the [U-TAE / L-TAE](https://github.com/VSainteuf/utae-paps) implementation (MIT).

## Layout

- `multisenge_utae/` — model, fusion of the two sensors, and the training loop
- `multisenge_seg/` — patch index, geographic split, class merge, and metrics

## Train

Point `--data-root` at a MultiSenGE folder that contains `s1/`, `s2/`, `labels/`, and `ground_reference/`. Build the index, train the decoder, then fine-tune.

```bash
python -m multisenge_seg.build_index --data-root /path/to/multisenge --out multisenge_seg/artifacts/patch_index.json

python -m multisenge_utae.train \
  --index multisenge_seg/artifacts/patch_index.json \
  --data-root /path/to/multisenge \
  --num-classes 6 \
  --mode head \
  --epochs 80 \
  --batch-size 2 \
  --out-dir multisenge_utae/checkpoints/run_c6_head

python -m multisenge_utae.train \
  --index multisenge_seg/artifacts/patch_index.json \
  --data-root /path/to/multisenge \
  --num-classes 6 \
  --mode full \
  --init-ckpt multisenge_utae/checkpoints/run_c6_head/best.pt \
  --epochs 80 \
  --batch-size 2 \
  --out-dir multisenge_utae/checkpoints/run_c6_full
```

Test the full checkpoint:

```bash
python -m multisenge_utae.train \
  --index multisenge_seg/artifacts/patch_index.json \
  --data-root /path/to/multisenge \
  --num-classes 6 \
  --eval-ckpt multisenge_utae/checkpoints/run_c6_full/best.pt \
  --eval-split test \
  --out-dir multisenge_utae/results/run_c6_full
```

## Split

| Split | Tiles | Patches |
| --- | --- | --- |
| train | 32UMV, 32ULU, 32TLT, 31UGQ, 31TFN, 31UFQ, 31UFR, 32ULV | ~3,369 |
| val | 31UFP, 31UGP | ~1,911 |
| test | 31UEQ | ~610 |
