'''
file_name,caption,vectorで構成されるcsvファイルを作成する。
99999枚まで対応。

・file_name: 画像ファイル名
・caption: 画像のキャプション
・vector: 画像の特徴量を表すベクトル（string型）
    →カンマ区切りのcsvにするため、vectorの中にカンマが含まれているとエラーが出る。
'''
import os
import csv
import re
import pandas as pd
from CLIP import clip
import torch
import argparse

def load_clip_model(device="cpu"):
    return clip.load("ViT-B/32", device=device)

def generate_caption_vector(caption, model, device):
    caption_inputs = clip.tokenize([caption]).to(device)
    caption_features = model.encode_text(caption_inputs)
    return ' '.join(map(str, caption_features.cpu().detach().numpy()[0]))

def rename_and_save_images(directory_path='images', csv_file_path='rename_results.csv', file_name_is_caption=True, force=False):
    if os.path.exists(csv_file_path) and not force:
        print("すでに完了しています。上書きの時は-fオプションを使ってください。")
        print("呼び出していないのにこのエラーが発生する場合は、カレントディレクトリにcsvファイルが存在していないかチェックして、存在する場合は削除してからリトライしてください。")
        return

    if force:
        print("Coming soon.")
        return

    counter = 1
    model, _ = load_clip_model(device="cuda" if torch.cuda.is_available() else "cpu")

    for filename in os.listdir(directory_path):
        match = re.match(r'(\d{5})\.png', filename)
        if match:
            counter = max(counter, int(match.group(1)) + 1)

    with open(csv_file_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Filename', 'Caption', 'Vector'])  # Add header

        for filename in os.listdir(directory_path):
            if filename.endswith('.png') and not re.match(r'\d{5}\.png', filename):
                new_filename = f"{counter:05d}.png"
                
                if file_name_is_caption:
                    caption = filename[:-4]
                else:
                    # Generate caption using some other method
                    pass
                
                vector = generate_caption_vector(caption, model, device="cuda" if torch.cuda.is_available() else "cpu")
                writer.writerow([new_filename, caption, vector])

                os.rename(
                    os.path.join(directory_path, filename),
                    os.path.join(directory_path, new_filename)
                )

                counter += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Rename and save images.')
    parser.add_argument('-f', '--force', action='store_true', help='Force renaming even if already done.')
    args = parser.parse_args()

    rename_and_save_images(force=args.force)
