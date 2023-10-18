# Assets_serch

## Image Serch
### Englich
This repository contains Python scripts for image search and management using OpenAI's CLIP (Contrastive Language–Image Pretraining) model. It allows users to:

1. Rename and save images with generated captions and vectors in a CSV file.
2. Perform image search based on text queries using precomputed vectors.

### Japanese
このリポジトリはテキストによる画像検索を行うためのpythonコードを管理しています。  
具体的にはOpenAIのCLIPとskleranのコサイン類似度を利用して特徴の近いものを抽出しています。  
ただし、使っている特徴量は画像の特徴ではなく、画像に付属しているCaptionと、検索クエリの類似度を計算しています。  
このモジュールは大きく２つの役割に分かれています:

1.　`make_csv.py`は画像ファイルのあるdir（file名=画像Caption）から、Captionの文字列をベクトル化し、csvに格納します。  
2.　`serch`は1で作ったcsvを利用して、クエリテキストとの類似度を計算し、類似度合いの高い画像のファイル名を提案するものです.


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

