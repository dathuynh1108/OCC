# NBD research presentation

`review.tex` is the editable source; `review.pdf` is the rendered 25-slide deck.

- Slides 3–8 introduce PatchCore, Kernel SVDD, Deep SVDD and DROCC.
- Slide 9 compares our actual paper-dataset reproduction measurements with the paper figures.
- Slide 10 compares all measured methods on the same MVTec AD test set.
- Slides 11–18 explain NBD. Slides 19–23 report the shared-feature controls, category results and diffusion ablation.

## Build

From this directory:

```sh
python3 generate_tables.py --mvtec-comparison ../../results/mvtec-native-full-4090-20260915
tectonic --keep-logs review.tex
```

The generator reads saved measurements only; it does not run training. It validates the 405 NBD category/seed rows, 905 native reproduction rows, and the MVTec comparison file hashes and test-ID/label equality checks before writing the six table fragments in `tables/`.

The MVTec comparison uses the completed seed-0 image-model measurements in `../../results/mvtec-native-full-4090-20260915/`. The separate NBD controls remain in `../../results/lean-nbd-5070-20260915/` and report three-seed means with sample SD.
