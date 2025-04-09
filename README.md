```markdown
# Assisted Fault Tree Construction using Sequential Recommendation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Code for the RESS paper:** "[Intelligent Fault Tree Construction for Automotive Diagnostics with Sequential Recommendation and Text Embeddings]"
<!-- Link to paper once available: [Link to Paper - e.g., DOI or arXiv] -->
<!-- Authors: [Your Name(s)] -->

## Description

This repository contains the code implementation for the research paper mentioned above, submitted to *Reliability Engineering & System Safety* (RESS). The project proposes and evaluates a framework that leverages sequential recommendation algorithms and text embeddings to assist engineers in constructing fault trees for automotive diagnostic applications. The goal is to improve the efficiency and consistency of FTA by providing data-driven suggestions for the next diagnostic node based on historical patterns and semantic context.

## Table of Contents

*   [Installation](#installation)
*   [Dataset](#dataset)
*   [Usage](#usage)
    *   [Preprocessing](#preprocessing)
    *   [Training](#training)
    *   [Evaluation](#evaluation)
*   [Reproducing Results](#reproducing-results)
*   [Models Implemented](#models-implemented)
*   [License](#license)
*   [Citation](#citation)
*   [Acknowledgements](#acknowledgements)
*   [Contact](#contact)

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/[your-username]/[your-repo-name].git
    cd [your-repo-name]
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    We primarily use the [RecBole](https://recbole.io/) framework and [Sentence Transformers](https://www.sbert.net/). Install dependencies using pip:
    ```bash
    pip install -r requirements.txt
    ```
    *   Python version: [e.g., 3.8+]
    *   PyTorch version: [e.g., 1.10+]
    *   RecBole version: [e.g., 1.1.1]
    *   Sentence Transformers version: [e.g., 2.2.2]
    *   [Other key libraries...]

## Dataset

*   **Data Source:** The experiments in the paper were conducted using a proprietary fault tree dataset from AVL List GmbH. Due to confidentiality, **this dataset cannot be shared publicly**.
*   **Data Format:** The code expects sequential data derived from fault trees in a specific format compatible with RecBole. Input data should generally be preprocessed into files (e.g., `.inter`, `.item`, `.user` - though users might not be relevant here) following RecBole's atomic file format guidelines.
    *   `.inter` file typically contains sequences: `[session_id]:token\t[item_id_sequence]:token_seq\t[timestamp_sequence]:float_seq` (timestamps might be optional/uniform).
    *   `.item` file maps `item_id:token` to `text:token` (for embeddings) or other features.
    <!-- Detail the specific columns/format your preprocessing script generates and the training scripts expect. -->
*   **Preprocessing:** See the [Preprocessing](#preprocessing) section below for details on how to transform raw fault tree data (if you had a sample format) into the required input format. The script `[your_preprocessing_script.py]` handles the transformation from [describe input format, e.g., JSON tree structure] to RecBole atomic files.

## Usage

### Preprocessing

If you have fault tree data in the expected input format [mention format again briefly], run the preprocessing script:

```bash
python [path/to/your_preprocessing_script.py] --input_path [path/to/raw/data] --output_path [path/to/processed/data/folder]
```

This script performs the following steps:
1.  Loads the fault tree data.
2.  Extracts sequential paths using the method described in the paper (Algorithm 1).
3.  Generates text embeddings for nodes using [mention embedding model, e.g., all-MiniLM-L6-v2] via Sentence Transformers and saves them [mention where/how, e.g., adds feature to .item file or saves separately].
4.  Formats the data into RecBole atomic files (`.inter`, `.item`) in the specified output directory.

### Training

Models are trained using RecBole's standard run script. Configuration files (`.yaml`) for different models and hyperparameters are provided in the `config/` directory.

To train a specific model (e.g., SASRec):

```bash
python run_recbole.py --model=[MODEL_NAME] --dataset=[DATASET_NAME] --config_files=[path/to/config/your_model_config.yaml]
```

*   `[MODEL_NAME]`: e.g., `SASRec`, `NARM`, `SRGNN`, `GRU4Rec`, `BERT4Rec`, `ItemKNN`, `SKNN`, `MarkovChain` etc.
*   `[DATASET_NAME]`: The name you assigned to your dataset during preprocessing (usually the name of the folder containing the atomic files).
*   `[path/to/config/your_model_config.yaml]`: Path to the specific YAML configuration file containing hyperparameters. Examples corresponding to the paper's experiments are provided.

<!-- Mention where trained models are saved (usually `saved/` by RecBole) -->

### Evaluation

Evaluation is typically performed automatically by RecBole at the end of training or can be run separately on a saved model. The framework calculates metrics like HR@k, NDCG@k, and MRR as defined in the paper. Results are logged to the console and potentially saved in output files within the `saved/` directory corresponding to the specific training run.

## Reproducing Results

To reproduce the main results reported in Table [Table number, e.g., Table 3] of the paper:

1.  Ensure the environment is set up correctly ([See Installation](#installation)).
2.  Preprocess your data into the required format using `[your_preprocessing_script.py]` (if applicable) or ensure your data is already in RecBole format. Place the processed data in the `dataset/[DATASET_NAME]` directory.
3.  Run the training script for each model using the corresponding configuration files from the `config/` directory. For example, to reproduce SASRec results for the 80/10/10 split:
    ```bash
    python run_recbole.py --model=SASRec --dataset=[DATASET_NAME] --config_files=config/sasrec_80_10_10_split.yaml
    ```
4.  The evaluation metrics will be printed at the end of each run. Compare these with the results reported in the paper.

*Note:* Due to the proprietary nature of the original dataset, exact replication requires access to comparable fault tree data. Results may vary depending on the characteristics of the dataset used. The provided configurations reflect the best hyperparameters found on the original dataset.

## Models Implemented

This repository includes implementations (primarily via RecBole) for the following models evaluated in the paper:

*   **Baselines:** Random, Most Popular
*   **Traditional:** Markov Chain (MC), FPMC
*   **Neural:** GRU4Rec, NARM, SASRec, BERT4Rec
*   **Graph-Based:** SR-GNN
*   **Embedding-Based:** ItemEmbeddingKNN (using precomputed text embeddings), SKNN

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

## Citation

If you find this work useful in your research, please consider citing our paper:

```bibtex
@article{[YourBibtexKey],
  author    = {[List of Authors]},
  title     = {Intelligent Fault Tree Construction for Automotive Diagnostics with Sequential Recommendation and Text Embeddings},
  journal   = {Reliability Engineering \& System Safety},
  year      = {[Year of Publication]},
  volume    = {[Volume]},
  pages     = {[Pages]},
  doi       = {[DOI]}
}
```

## Acknowledgements

*   This work was partially supported by the FFG HybridAir project #FO999902654.
*   This work was supported by the "DDIA" COMET Module within the COMET – Competence Centers for Excellent Technologies Programme, funded by the Austrian Federal Ministry for Transport, Innovation and Technology (bmvit), the Austrian Federal Ministry for Digital and Economic Affairs (bmdw), FFG, SFG, and partners from industry and academia. The COMET Programme is managed by FFG.
*   We thank AVL List GmbH for providing the dataset and domain expertise.
*   [Any other acknowledgements]

## Contact

For questions about the paper or code, please contact [Your Name] at [your.email@example.com] or open an issue in this repository.
```
