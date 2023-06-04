from nomic import atlas
import numpy as np
import torch
from Bio.SeqUtils.ProtParam import ProteinAnalysis


sstrans = str.maketrans('EMALKNPGSDVIYFWLT', 'HHHHHTTTTTSSSSSSS')

def protein_info(seq: str) -> dict[str]:
    pa = ProteinAnalysis(seq.replace('X', '').replace('B', ''))
    return {
        'sequence': seq,
        'molecular_weight': pa.molecular_weight(),
        'aromaticity': pa.aromaticity(),
        'instability': pa.instability_index(),
        'isoelectric_point': pa.isoelectric_point(),
        'secondary_structure': seq.translate(sstrans),
        'gravy': pa.gravy(),
    }


if __name__ == '__main__':
    with open('aa.txt') as f:
        sequences = [line[:-1] for line in f]
    embeddings = torch.load('structural.pt')

    project = atlas.map_embeddings(
        embeddings=embeddings.numpy(),
        data=[protein_info(seq) for seq in sequences],
        name='svm',
        id_field='sequence',
        reset_project_if_exists=True,
        build_topic_model=True,
    )
    print(project)