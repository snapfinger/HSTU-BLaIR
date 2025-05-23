### HSTU-BLaIR: Lightweight Contrastive Text Embedding for Generative Recommender

Recent advances in recommender systems have underscored the complementary strengths of generative modeling and pretrained language models. We propose **HSTU-BLaIR**, a hybrid framework that augments the Hierarchical Sequential Transduction Unit (HSTU)-based generative recommender with BLaIR, a lightweight contrastive text embedding model. This integration enriches item representations with semantic signals from textual metadata while preserving HSTU’s powerful sequence modeling capabilities.

We evaluate HSTU-BLaIR on two e-commerce datasets: three subsets from the Amazon Reviews 2023 dataset and the Steam dataset. We compare its performance against both the original HSTU-based recommender and a variant augmented with embeddings from OpenAI’s state-of-the-art \texttt{text-embedding-3-large} model. Despite the latter being trained on a substantially larger corpus with significantly more parameters, our lightweight BLaIR-enhanced approach—pretrained on domain-specific data—achieves better performance in nearly all cases. Specifically, HSTU-BLaIR outperforms the OpenAI embedding-based variant on all but one metric, where it is marginally lower, and matches it on another. These findings highlight the effectiveness of contrastive text embeddings in compute-efficient recommendation settings.

📄 The full technical report can be found [here](https://arxiv.org/pdf/2504.10545).

---

## 🚀 Getting Started

Install the required Python packages with ```pip3 install -r requirements.txt```.

**HSTU-BLaIR** is built on top of [the HSTU-based generative recommender (commit `ece916f`)](https://github.com/facebookresearch/generative-recommenders/tree/ece916f) and extends it with additional modules for integrating textual embeddings. It has been tested on Ubuntu 22.04 with Python 3.9, CUDA 12.6, and a single NVIDIA RTX 4090 GPU.

---

## 🧪 Experiments

To reproduce the experiments from our [technical report](https://arxiv.org/pdf/2504.10545), follow these steps:

### 1. Download and preprocess the data

```bash
mkdir -p tmp/ && python3 preprocess_public_data.py
```

To use OpenAI `text-embedding-3-large` embeddings instead of BLaIR, update the corresponding line in `preprocess_public_data.py` as follows:

```python
text_embedding_model = "blair"
```

to

```python
text_embedding_model = "openai"
```

Make sure you have correctly set up your OpenAI API credentials if you choose this option.


### 2. Run the model

```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py --gin_config_file=configs/amzn23_game/hstu-sampled-softmax-n512-blair.gin --master_port=12345
```

You can find other configuration files in the configs/ directory for different settings and domains.

---

## 📊 Results
**Evaluation metrics on Video Games, Office Products, and Musical Instruments of Amazon Reviews 2023 dataset and Steam dataset**  

With the provided `.gin` files, you should be able to reproduce the following evaluation results within a small margin of variability. Minor differences may arise due to inherent non-determinism in model initialization, data loading, and/or hardware-level operations.

Each cell shows absolute values (top row) and percentage improvements over HSTU / SASRec (bottom row). Best values per column are bolded. 

*HSTU-OpenAI (TE3L)* denotes the HSTU model augmented with text embeddings generated via OpenAI’s `text-embedding-3-large` API. 

### Video Games

| Model               | HR@10     | HR@50     | HR@200    | NDCG@10   | NDCG@200 | MRR      |
|---------------------|-----------|-----------|-----------|-----------|----------|----------|
| SASRec              | .1028     | .2317     | .3941     | .0573     | .1097    | .0518    |
|                     | —         | —         | —         | —         | —        | —        |
| HSTU                | .1315     | .2765     | .4565     | .0741     | .1327    | .0658    |
|                     | (+28%)    | (+19.3%)  | (+15.8%)  | (+29.3%)  | (+21.0%) | (+27.1%) |
| HSTU-OpenAI (TE3L)  | .1328     | .2821     | .4645     | .0742     | .1341    | .0658    |
|                     | (+1.0% / +29.2%) | (+2.0% / +21.8%) | (+1.8% / +17.9%) | (+0.1% / +29.5%) | (+1.1% / +22.2%) | (0.0% / +27.0%) |
| **HSTU-BLaIR**      | **.1353** | **.2852** | **.4684** | **.0760** | **.1361**| **.0674**|
|                     | (+2.9% / +31.6%) | (+3.1% / +23.1%) | (+2.6% / +18.9%) | (+2.6% / +32.6%) | (+2.6% / +24.1%) | (+2.4% / +30.1%) |

### Office Products

| Model               | HR@10     | HR@50     | HR@200    | NDCG@10   | NDCG@200 | MRR      |
|---------------------|-----------|-----------|-----------|-----------|----------|----------|
| SASRec              | .0281     | .0668     | .1331     | .0153     | .0335    | .0143    |
|                     | —         | —         | —         | —         | —        | —        |
| HSTU                | .0395     | .0880     | .1649     | .0223     | .0443    | .0207    |
|                     | (+40.6%)  | (+31.7%)  | (+23.9%)  | (+45.8%)  | (+32.2%) | (+44.8%) |
| HSTU-OpenAI (TE3L)  | .0477     | .1050     | .1940     | .0269     | .0526    | .0247    |
|                     | (+20.8% / +69.8%) | (+19.3% / +57.2%) | (+17.6% / +45.8%) | (+20.6% / +75.8%) | (+18.7% / +57.0%) | (+19.3% / +72.7%) |
| **HSTU-BLaIR**      | **.0484** | **.1068** | **.1946** | **.0271** | **.0529**| **.0248**|
|                     | (+22.5% / +72.2%) | (+21.4% / +59.9%) | (+18.0% / +46.2%) | (+21.5% / +77.1%) | (+19.4% / +57.9%) | (+19.8% / +73.4%) |

### Musical Instruments

| Model               | HR@10     | HR@50     | HR@200    | NDCG@10   | NDCG@200 | MRR      |
|---------------------|-----------|-----------|-----------|-----------|----------|----------|
| SASRec              | .0643     | .1488     | .2784     | .0356     | .0731    | .0326    |
|                     | —         | —         | —         | —         | —        | —        |
| HSTU                | .0700     | .1599     | .2910     | .0392     | .0783    | .0359    |
|                     | (+8.9%)   | (+7.5%)   | (+4.5%)   | (+10.1%)  | (+7.1%)  | (+10.1%) |
| HSTU-OpenAI (TE3L)  | .0708     | .1635     | .3005     | .0393     | .0798    | .0360    |
|                     | (+1.1% / +10.1%) | (+2.3% / +9.9%) | (+3.3% / +7.9%) | (+0.3% / +10.4%) | (+1.9% / +9.2%) | (+0.3% / +10.4%) |
| **HSTU-BLaIR**      | **.0733** | **.1681** | **.3066** | **.0406** | **.0818**| **.0371**|
|                     | (+4.7% / +14.0%) | (+5.1% / +13.0%) | (+5.4% / +10.1%) | (+3.6% / +14.0%) | (+4.5% / +11.9%) | (+3.3% / +13.8%) |

### Steam

| Model               | HR@10     | HR@50     | HR@200    | NDCG@10   | NDCG@200 | MRR      |
|---------------------|-----------|-----------|-----------|-----------|----------|----------|
| SASRec              | .0881     | .2277     | .4312     | .0462     | .1068    | .0426    |
|                     | —         | —         | —         | —         | —        | —        |
| HSTU                | .1038     | .2575     | .4704     | .0544     | .1195    | .0492    |
|                     | (+17.8%)  | (+13.1%)  | (+9.1%)   | (+17.8%)  | (+11.9%) | (+15.5%) |
| HSTU-OpenAI (TE3L)  | .1089     | .2657     | .4806     | .0579     | **.1241**| **.0525**|
|                     | (+4.9% / +23.6%) | (+3.2% / +16.7%) | (+2.2% / +11.5%) | (+6.4% / +25.3%) | (+3.8% / +16.2%) | (+6.7% / +23.2%) |
| **HSTU-BLaIR**      | **.1100** | **.2668** | **.4812** | **.0581** | **.1241**| .0523    |
|                     | (+6.0% / +24.9%) | (+3.6% / +17.2%) | (+2.3% / +11.6%) | (+6.8% / +25.8%) | (+3.8% / +16.2%) | (+6.3% / +22.8%) |

## 📚 Citation

If you use HSTU-BLaIR in your work, please cite:

```bibtex
@article{hstublair2025,
  title     = {Integrating Textual Embeddings from Contrastive Learning with Generative Recommender for Enhanced Personalization},
  author    = {Yijun Liu},
  journal   = {arXiv preprint arXiv:2504.10545},
  year      = {2025},
  url       = {https://arxiv.org/abs/2504.10545}
}
```

## 📬 Contact

If you encounter any problems or have questions, feel free to [open an issue](https://github.com/snapfinger/HSTU-BLaIR/issues) or email yijunl [at] usc [dot] edu.

## License
This codebase is Apache 2.0 licensed, as found in the [LICENSE](LICENSE) file.
