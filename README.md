# denoising

The default is the 8.23M model: six layers, width 256, eight heads, 32 global slots,
and learned pairwise mass attention with 32 strong-peak landmarks. The previous
3.16M architecture is commented in the training config for reference.

```bash
python split_data_xl.py --config config/train_xl.yml
python -u -m src.training.train_xl --config config/train_xl.yml
python -u -m src.inference.run_xl --config config/infer_xl.yml
```

Edit the canonical settings in `configs/train_xl.yml` and the input MGF paths in
`configs/infer_xl.yml`. Split directories and training output directories must be
new for fresh experiments. Existing XL splits can be reused for training.
