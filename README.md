# Korean License Plate Generator

A synthetic Korean license plate image generator for license plate recognition (LPR/OCR) research and model pretraining.

This project generates Korean license plate ROI images with character-level bounding box annotations. It is designed to create large-scale synthetic data for lightweight Korean LPR models, especially when real unmasked Korean license plate data is difficult to obtain due to privacy restrictions.

## Features

- Generate multiple Korean license plate types
- Support character-level bounding box annotations
- Support Korean digits, Hangul vehicle class characters, and region names
- Support realistic degradation and augmentation
- Output data in a simple image + text annotation format
- Suitable for OCR/LPR pretraining and domain adaptation experiments

## Supported Plate Types

| Type | Description |
|---|---|
| A | Standard white single-line plate |
| B | Old white two-line plate |
| C | Old yellow two-line plate |
| D | Region-based horizontal plate |
| E | New white plate with blue security mark |
| F | EV plate |
| G | Old green plate |
| H | New green long plate |

## Data Format

The generated dataset is saved as image files and text annotation files.

Example:

```text
output/
├── labels.names
├── chunk_00/
│   ├── 12가3456_000001.jpg
│   ├── 12가3456_000001.txt
│   ├── 123가4567_000002.jpg
│   └── 123가4567_000002.txt
└── chunk_01/
```

Each `.txt` file contains one bounding box per character:

```text
x1 y1 x2 y2
x1 y1 x2 y2
...
```

The license plate text is stored in the image filename.

Example:

```text
12가3456_000001.jpg
```

means the plate text is:

```text
12가3456
```

## Character Labels

The `labels.names` file contains the OCR character vocabulary.

The first line is reserved for CTC blank.

Example:

```text

0
1
2
3
...
가
나
다
...
서울
부산
경기
...
```

When training a CTC-based OCR model, make sure the number of classes matches the number of lines in `labels.names`.

## Usage

Generate a small test dataset:

```bash
python generate_all.py \
  --total 1000 \
  --workers 1 \
  --chunk-size 1000 \
  --out ./output_debug
```

Generate a large dataset:

```bash
python generate_all.py \
  --total 300000 \
  --workers 32 \
  --chunk-size 10000 \
  --out ./output_30w
```

## Arguments

| Argument | Description |
|---|---|
| `--total` | Total number of images to generate |
| `--workers` | Number of parallel worker processes |
| `--chunk-size` | Number of samples per output chunk |
| `--out` | Output directory |
| `--asset-dir` | Asset directory path |

## Asset Structure

Expected asset directory structure:

```text
assets/
├── nums/
├── chars/
├── chars_truck/
├── region1/
├── region2/
└── plates/
    ├── type_a/
    ├── type_b/
    ├── type_c/
    ├── type_d/
    ├── type_e/
    ├── type_f/
    ├── type_g/
    └── type_h/
```

## Augmentation

The generator includes several image degradation and augmentation methods:

- Brightness and contrast variation
- Random crop and padding
- Low-resolution simulation
- Motion blur
- Gaussian blur
- JPEG compression
- Sensor noise
- Local glare
- Print-like artifacts
- Slight affine and perspective distortion

These augmentations are intended to reduce the gap between clean synthetic images and real-world parking lot camera images.

## Recommended Workflow

This generator is intended mainly for pretraining.

Recommended training pipeline:

```text
Synthetic data pretraining
        ↓
Real ROI fine-tuning
        ↓
Real validation set evaluation
```

Synthetic images are useful for learning:

- Korean plate layouts
- Character categories
- CTC alignment
- Basic OCR behavior
- Rare plate formats

However, final deployment performance should always be validated on real cropped license plate images.

## Notes

Synthetic data cannot fully replace real data. Real Korean license plate images often include:

- Camera blur
- JPEG artifacts
- Reflection
- Low resolution
- Dirty plate surfaces
- Old plate materials
- Different fonts
- Different lighting conditions

For production use, fine-tuning with real license plate ROI images is strongly recommended.

## License

Please add your project license here before publishing.

Example options:

- MIT License
- Apache License 2.0
- Custom research-only license

Make sure all image assets used in `assets/` are allowed to be redistributed before publishing this repository.

## Disclaimer

This project is intended for research and development of license plate recognition systems.  
Please follow local privacy laws and data protection regulations when collecting, generating, or using vehicle license plate data.
## Reference：
https://github.com/qjadud1994/Korean-license-plate-Generator
https://github.com/yakhyo/korean-license-plate-generator
https://github.com/Usmankhujaev/KoreanCarPlateGenerator

```