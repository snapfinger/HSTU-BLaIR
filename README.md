### Integrating Textual Embeddings from Contrastive Learning with a Generative Recommender for Enhanced Personalization

We present **HSTU-BLaIR**, a hybrid framework that augments the Hierarchical Sequential Transduction Unit (HSTU) generative recommender with BLaIR—a contrastive text embedding model. This integration enriches item representations with semantic signals from textual metadata while preserving HSTU's powerful sequence modeling capabilities.

We evaluate our method on two domains from the Amazon Reviews 2023 dataset, comparing it against the original HSTU and a variant that incorporates embeddings from OpenAI’s state-of-the-art `text-embedding-3-large` model. While the OpenAI embedding model is likely trained on a substantially larger corpus with significantly more parameters, our lightweight BLaIR-enhanced approach—pretrained on domain-specific data—consistently achieves better performance, highlighting the effectiveness of contrastive text embeddings in compute-efficient settings.

📄 The full technical report can be found [here](https://arxiv.org/pdf/2504.10545).

---

## 🚀 Getting Started

Install the required Python packages with ```pip3 install -r requirements.txt```.

**HSTU-BLaIR** is built on top of [the HSTU-based generative recommender (commit `ece916f`)](https://github.com/facebookresearch/generative-recommenders/tree/ece916f) and extends it with additional modules for integrating textual embeddings. It has been tested on Ubuntu 22.04 with Python 3.9, CUDA 12.6, and a single NVIDIA RTX 4090 GPU.

---

## 🧪 Experiments

To reproduce the experiments from our [technical report](https://arxiv.org/pdf/2504.10545) on the Amazon Reviews 2023 dataset, follow these steps:

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
**Evaluation metrics on Video Games and Office Products datasets**  

With the provided `.gin` files, you should be able to reproduce the following evaluation results within a small margin of variability. Minor differences may arise due to inherent non-determinism in model initialization, data loading, and/or hardware-level operations.

Each cell shows absolute values (top row) and percentage improvements over HSTU / SASRec (bottom row). Best values per column are bolded. 

*HSTU-OpenAI (TE3L)* denotes the HSTU model augmented with text embeddings generated via OpenAI’s `text-embedding-3-large` API, queried on or before April 7, 2025. 

#### Video Games

| Model                  | HR@10     | HR@50     | HR@200    | NDCG@10   | NDCG@200  |
|------------------------|-----------|-----------|-----------|-----------|-----------|
| SASRec                | 0.1028    | 0.2317    | 0.3941    | 0.0573    | 0.1097    |
|                        | —         | —         | —         | —         | —         |
| HSTU                  | 0.1315    | 0.2765    | 0.4565    | 0.0741    | 0.1327    |
|                        | (+28%)    | (+19%)    | (+16%)    | (+29%)    | (+21%)    |
| HSTU-OpenAI (TE3L)     | 0.1328    | 0.2821    | 0.4645    | 0.0742    | 0.1341    |
|                        | (+1.0% / +29%) | (+2.0% / +22%) | (+1.8% / +18%) | (+0.1% / +30%) | (+1.1% / +22%) |
| **HSTU-BLaIR**         | **0.1353**| **0.2852**| **0.4684**| **0.0760**| **0.1361**|
|                        | (+2.9% / +32%) | (+3.1% / +23%) | (+2.6% / +19%) | (+2.6% / +33%) | (+2.6% / +24%) |

#### Office Products

| Model                  | HR@10     | HR@50     | HR@200    | NDCG@10   | NDCG@200  |
|------------------------|-----------|-----------|-----------|-----------|-----------|
| SASRec                | 0.0281    | 0.0668    | 0.1331    | 0.0153    | 0.0335    |
|                        | —         | —         | —         | —         | —         |
| HSTU                  | 0.0395    | 0.0880    | 0.1649    | 0.0223    | 0.0443    |
|                        | (+41%)    | (+32%)    | (+24%)    | (+46%)    | (+32%)    |
| HSTU-OpenAI (TE3L)     | 0.0477    | 0.1050    | 0.1940    | 0.0269    | 0.0526    |
|                        | (+20.8% / +70%) | (+19.3% / +57%) | (+17.6% / +46%) | (+20.6% / +76%) | (+18.7% / +57%) |
| **HSTU-BLaIR**         | **0.0484**| **0.1068**| **0.1946**| **0.0271**| **0.0529**|
|                        | (+22.5% / +72%) | (+21.4% / +60%) | (+18.0% / +46%) | (+21.5% / +77%) | (+19.4% / +58%) |



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
