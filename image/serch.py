'''
make_csv.pyで作成したcsvファイルを使って画像検索を行うpython_file
'''
import pandas as pd
from CLIP import clip
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class ImageSearcher:
    def __init__(self, model_path="ViT-B/32", csv_path='rename_results.csv'):
        self.model, _ = clip.load(model_path)
        self.model.eval()
        self.data = self._load_csv(csv_path)
        
    def _load_csv(self, csv_path):
        data = pd.read_csv(csv_path)
        data['Vector'] = data['Vector'].apply(lambda x: np.array(list(map(float, x.split()))))
        return data
    
    def vectorize_text(self, text):
        text_input = clip.tokenize([text]).cuda()
        text_features = self.model.encode_text(text_input)
        return text_features.cpu().detach().numpy()

    def search_images(self, query, top_n=3):
        query_vector = self.vectorize_text(query)
        data_vectors = np.vstack(self.data['Vector'])
        similarities = cosine_similarity(query_vector, data_vectors).squeeze()
        top_indices = similarities.argsort()[-top_n:][::-1]
        top_images = self.data['Filename'].iloc[top_indices].tolist()
        return top_images

if __name__ == "__main__":
    searcher = ImageSearcher()
    query = input('検索したいキーワードを入力してください: ')
    top3_images = searcher.search_images(query)
    print(top3_images)
