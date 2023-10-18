# CLIP Image Search and Management

## Description

This repository contains Python scripts for image search and management using OpenAI's CLIP (Contrastive Language–Image Pretraining) model. It allows users to:

1. Rename and save images with generated captions and vectors in a CSV file.
2. Perform image search based on text queries using precomputed vectors.

## Requirements

- Python 3.10
- pandas
- torch
- CLIP
- scikit-learn

To install the required packages, run:

```bash
pip install -r requirements.txt
```

## How to Use

### Rename and Save Images

To rename and save images, run the following command:

```bash
python make_csv.py
```

To override the existing CSV, use the `-f` or `--force` flag:

```bash
python make_csv.py -f
```

> Note: Implementation of `-f` option is coming soon.

### Image Search

To perform an image search, run the following command:

```bash
python search_images.py "Your text query here"
```

## Notes

- For better performance, ensure you have CUDA support for PyTorch if available.

