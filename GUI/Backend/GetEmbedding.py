# Utilities
############################################################################
# Imports
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle


############################################################################
# Function to read the embedding_dataset from the "total_embedding.pickle"
def creatEmbedding():
    with open("C:\\Users\\balln\\Desktop\\PART_PythonGUI\\data_base\\total_embedding.pickle", 'rb') as f:
        embedding_dataset = pickle.load(f)
    return embedding_dataset


############################################################################
# Returns 2 embedding vectors one of the author of the imposter text and one for William Shakespeare
def twoEmbeddings(embedding_dataset, imposterText):
    # Find The embedding of the imposter book that was chosen
    filtered_imposter = embedding_dataset[embedding_dataset['book'] == imposterText]
    embedding_imposter = None
    if not filtered_imposter.empty:
        # Access the embedding value of the first matching row
        embedding_imposter = filtered_imposter['embedding'].iloc[0]

    # Get The embedding of the first book matched to be written by William Shakespeare
    filtered_shake = embedding_dataset[embedding_dataset['author'] == 'Shakespeare']
    embedding_shake = None
    if not filtered_shake.empty:
        embedding_shake = filtered_shake['embedding'].iloc[0]
    return embedding_shake, embedding_imposter


#################################################################################################
# Return cosine Similarity between 2 embeddings
def cosine_similarity_percentage(embed1, embed2):
    similarity = cosine_similarity(embed1.reshape(1, -1), embed2.reshape(1, -1))
    similarity_percentage = np.clip(similarity[0][0], -1, 1) * 100
    return similarity_percentage
