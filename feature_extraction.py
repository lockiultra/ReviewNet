import sys

from models import GPTFeatureExtraction
import pandas as pd

def main(path_to_data: str):
    fe = GPTFeatureExtraction()
    data = pd.read_csv(path_to_data)
    print(data)
    X = fe(data['text'].values)
    pd.DataFrame(X).to_csv('./data_features.csv')

if __name__ == '__main__':
    path_to_data = sys.argv[1]
    main(path_to_data)
