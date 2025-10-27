import numpy as np
from sentence_transformers import SentenceTransformer


class SBERT(object):
    def __init__(self, model_name='sentence-transformers/paraphrase-xlm-r-multilingual-v1'):
        super(SBERT, self).__init__()
        # Load the SentenceTransformer model (multilingual)
        self.model = SentenceTransformer(model_name)

    def semantic_sim(self, sents1, sents2):
        """
        :param sents1: list of sentence strings
        :param sents2: list of sentence strings
        :return: [sim_scores], where sim_scores is a NumPy array of shape (N,)
                 following the same structure as the old code
        """
        # Encode both sets of sentences
        e1 = self.model.encode(sents1)  # shape (N, D)
        e2 = self.model.encode(sents2)  # shape (N, D)
        
        # L2-normalize each embedding so dot product = cosine similarity
        e1 = e1 / np.linalg.norm(e1, axis=1, keepdims=True)
        e2 = e2 / np.linalg.norm(e2, axis=1, keepdims=True)
        
        # Cosine similarity is just the row-wise dot product now
        cos_sim = np.sum(e1 * e2, axis=1)  # shape (N,)
        
        # Clip to avoid numerical issues outside [-1, 1]
        cos_sim = np.clip(cos_sim, -1.0, 1.0)
        
        # Same transformation as your old code: 1.0 - arccos(cos_sim)
        # If cos_sim=1.0 => score = 1.0
        # If cos_sim=-1.0 => score = 1.0 - π  (~ -2.14)
        sim_scores = 1.0 - np.arccos(cos_sim)

        # Return in the same shape as your TF code: a list with one array
        return [sim_scores]
