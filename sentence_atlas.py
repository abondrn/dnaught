from nomic import atlas
import pandas as pd


def load_sentence_embeddings():
    df = pd.read_csv('SVM/protein_whole_text_embedding.csv')
    return df.to_numpy().T


if __name__ == '__main__':
    # loads GO embeddings 
    emb = load_sentence_embeddings()

    project = atlas.map_embeddings(
        embeddings=emb,
        #data=data,
        name='svm-sentence',
        #id_field='sequence',
        reset_project_if_exists=True,
        build_topic_model=True,
        is_public=True,
    )
    print(project)