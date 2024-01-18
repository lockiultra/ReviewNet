import torch
from transformers import pipeline
from tqdm import tqdm

class GPTFeatureExtraction:
    def __init__(self, embedding_size=768, model='ai-forever/rugpt3small_based_on_gpt2'):
        self.embedding_size = embedding_size
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.feature_extraction_pipeline = pipeline(task='feature-extraction', model=model, device=self.device)

    def __mean_pooling(self, model_output):
        emb_sum = torch.zeros(self.embedding_size)
        for emb in model_output:
            emb_sum += torch.FloatTensor(emb)
        return emb_sum / len(model_output)
    
    def __call__(self, data):
        fe_output = []
        for text in tqdm(data):
            features = self.feature_extraction_pipeline(text)
            features_pooled = self.__mean_pooling(features[0])
            fe_output.append(features_pooled)
        return torch.stack(fe_output)
