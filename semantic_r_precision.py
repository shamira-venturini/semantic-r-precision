from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch
from nltk.stem.porter import PorterStemmer

# Global model and stemmer to avoid reloading on every call
_MODEL_CACHE = {}
_STEMMER = PorterStemmer()


def _get_model(model_name_or_path):
    if model_name_or_path not in _MODEL_CACHE:
        _MODEL_CACHE[model_name_or_path] = SentenceTransformer(model_name_or_path)
    return _MODEL_CACHE[model_name_or_path]


def _stem_phrase_list(phrases):
    return [" ".join([_STEMMER.stem(token) for token in phrase.lower().split()]) for phrase in phrases]


def calculate_sem_r_p(predictions, references, k=3, model_name_or_path='uclanlp/keyphrase-mpnet-v1'):
    """
    Calculates Semantic R-Precision (SemR-p).

    Args:
        predictions (list of str): Ordered list of predicted keyphrases.
        references (list of str): List of reference keyphrases.
        k (int): Number of top similar references to consider for semantic score.
        model_name_or_path (str): Path or Hugging Face Hub name of the Sentence Transformer model.

    Returns:
        float: The SemR-p score.
    """
    model = _get_model(model_name_or_path)

    stemmed_refs = _stem_phrase_list(references)
    num_refs = len(stemmed_refs)

    if num_refs == 0:
        return 0.0

    top_r_predictions = predictions[:num_refs]
    if not top_r_predictions:
        return 0.0

    stemmed_top_r_preds = _stem_phrase_list(top_r_predictions)

    individual_scores = []
    non_match_preds_for_embedding = []
    non_match_indices_in_top_r = []

    for i, pred_stemmed in enumerate(stemmed_top_r_preds):
        if pred_stemmed in stemmed_refs:
            individual_scores.append(1.0)
        else:
            individual_scores.append(None)  # Placeholder for semantic score
            non_match_preds_for_embedding.append(top_r_predictions[i])  # Use original for embedding
            non_match_indices_in_top_r.append(i)

    if non_match_preds_for_embedding:
        pred_embeddings = model.encode(non_match_preds_for_embedding)
        ref_embeddings = model.encode(references)  # Use original references for embedding
        sim_matrix = util.cos_sim(pred_embeddings, ref_embeddings)

        for i, original_idx in enumerate(non_match_indices_in_top_r):
            top_k_sims = torch.topk(sim_matrix[i], min(k, num_refs)).values
            individual_scores[original_idx] = top_k_sims.mean().item()

    for i in range(len(individual_scores)):
        if individual_scores[i] is None:
            individual_scores[i] = 0.0

    return np.mean(individual_scores) if individual_scores else 0.0