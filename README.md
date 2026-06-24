# Semantic R-Precision (SemR-p)

Semantic R-Precision (SemR-p) is a novel metric designed to jointly evaluate the semantic relevance and ranking quality of predicted keyphrases against reference keyphrases. It builds upon the R-Precision framework by incorporating embedding-based semantic similarity for non-exact matches, aiming to provide a more holistic assessment of keyphrase quality.

This implementation is intended for standalone use and was developed alongside our research paper. We also aim to contribute SemR-p to the main [KPEval toolkit](https://github.com/uclanlp/KPEval).

**For a detailed explanation of the metric, its motivation, and comprehensive evaluation, please refer to our paper:**
[Meaning in Order, Order in Meaning: Semantic R-precision for Keyphrase Evaluation](
https://doi.org/10.48550/arXiv.2606.07057) (arXiv Preprint)

## Citation
If you use SemR-p in your research, please cite our paper:

```bibtex
@misc{semantic-r-precision,
  doi = {10.48550/ARXIV.2606.07057},
  url = {https://arxiv.org/abs/2606.07057},
  author = {Venturini,  Shamira and Kinkel,  Steffen},
  keywords = {Information Retrieval (cs.IR),  Computation and Language (cs.CL),  FOS: Computer and information sciences,  FOS: Computer and information sciences},
  title = {Meaning in Order,  Order in Meaning: Semantic R-precision for Keyphrase Evaluation},
  publisher = {arXiv},
  year = {2025},
  copyright = {Creative Commons Attribution 4.0 International}
}

```

## Features
*   Evaluates top-R predictions, making it rank-aware and adaptive to the number of reference keyphrases.
*   Prioritises exact stem matches for computational efficiency
*   Utilises transformer-based embeddings (via Sentence Transformers) for robust semantic similarity scoring of non-exact matches.
*   Considers the top-*k* most similar references for semantic scoring, allowing for nuanced evaluation.

---

## Installation

1.  **Clone the repository**
``` bash
git clone https://github.com/your-username/semantic-r-precision.git
cd semantic-r-precision
```

2. **Set up a virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
3. **Install the required dependencies**:
```
pip install -r requirements.txt
```



## Usage

``` python
from evaluation_metrics import semantic_r_precision

# Example data
reference_keyphrases = [
    ["machine learning", "artificial intelligence", "deep learning"],
    ["natural language processing", "text mining", "NLP"]
]
predicted_keyphrases = [
    ["deep learning", "machine learning", "AI"],
    ["NLP", "text analytics", "language understanding"]
]

# Compute Semantic R-Precision with default k=3
score = semantic_r_precision(reference_keyphrases, predicted_keyphrases. k=3, model_name_or_path='uclanlp/keyphrase-mpnet-v1')

print(f"Semantic R-Precision: {score:.4f}")
```

## Customisation

* *k*: you can customise parameter k. In our [paper](https://www.researchgate.net/publication/391552955_Meaning_in_Order_Order_in_Meaning_Semantic_R-precision_for_Keyphrase_Evaluation)
we go through the reasons why we chose *k*=3 as default and what happens when you change it to *k*=1.
* embedding model: change it to your preferred one

