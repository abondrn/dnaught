import os 
import tempfile
from zipfile import ZipFile

from nomic import atlas
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd


class GOEmbeddingsDataset(torch.utils.data.Dataset):
    def __init__(self, go_id2vec_dict):
        self.embeddings = list(go_id2vec_dict.items())
        
    def __len__(self):
        return len(self.embeddings)
      
    def __getitem__(self, index): 
        go_id, vec = self.embeddings[index]
        vec = np.array([vec])
        return go_id, torch.from_numpy(vec)


def load_go_embeddings(path = "SVM/go_embeddings_20k.zip"):
    emb = []
    data = []

    # Unzip the zip folder to a temporary directory       
    with ZipFile(path, "r") as zip_file:                
        # Get list of .npy files
        for f in tqdm(zip_file.namelist()):
            if f.endswith('.npy'):
                # Load each .npy file
                vec = np.load(zip_file.open(f))
                emb.append(vec)
                id = f.split('/')[1].split('.')[0]
                data.append(go_terms[id])
            
    return emb, data


df = pd.read_csv('SVM/Uniprot-extracted_comments_and_GO_terms.csv', index_col=0)
go_terms = {r['primaryAccession']: r for r in df.fillna('').to_dict(orient='records')}

# Initialize the dataset       
#go_dataset = GOEmbeddingsDataset(go_id2vec_dict)

# Initialize the dataloader     
#go_loader = DataLoader(
#    go_dataset, 
#    batch_size=4,  
#    shuffle=True
#)

if __name__ == '__main__':
    # loads GO embeddings 
    emb, data = load_go_embeddings()
    print(data[0])

    project = atlas.map_embeddings(
        embeddings=np.array(emb),
        data=data,
        name='svm-go',
        #id_field='sequence',
        reset_project_if_exists=True,
        build_topic_model=True,
        is_public=True,
    )
    print(project)