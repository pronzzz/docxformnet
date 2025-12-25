# FormNet Developer Guide

## Project Structure

```
docxformnet/
├── data/               # Scripts and utilities for data loading
│   ├── download_datasets.py  # Download CORD/FUNSD
│   └── dataset.py            # PyTorch Dataset implementation
├── preprocessing/      # Data preprocessing pipeline
│   └── pipeline.py           # Normalization, resizing, graph construction
├── tokenization/       # Embedding modules
│   ├── text_embedding.py     # DistilBERT wrapper
│   ├── spatial_embedding.py  # Coordinate embeddings
│   └── combined_embedding.py # Fusion layer
├── rich_attention/     # Core interaction mechanism
│   └── rich_attention.py     # Spatial-aware self-attention
├── gcn_super_tokens/   # Context refinement
│   └── gcn.py                # Graph Convolutional Network layers
├── model/              # Main model assembly
│   └── formnet.py            # FormNet class
├── utils/              # Helpers
│   ├── collator.py           # Batch processing and padding
│   └── visualize.py          # Visualization tools
├── train.py            # Training loop
└── evaluate.py         # Evaluation script
```

## Dataset Details
This project is configured to work with the **CORD** (Consolidated Receipt Dataset) and **FUNSD** (Form Understanding in Noisy Scanned Documents) datasets.
- **CORD**: Contains receipts with fields like 'menu.nm', 'total.price', etc.
- **FUNSD**: Contains generic forms with key-value pairs and headers.

The `data/dataset.py` script automatically handles the JSON structure of these datasets provided by Hugging Face `datasets` library.

## Component Deep Dive

### 1. Preprocessing & Graph Construction
We use a **K-Nearest Neighbors (KNN)** approach to build a spatial graph for each document.
- **Nodes**: Tokens (subwords).
- **Edges**: Connect tokens that are spatially close.
- **Adjacency Matrix**: A binary or weighted matrix `[MaxLen, MaxLen]` passed to the GCN.

### 2. Rich Attention
The `RichAttention` module extends standard multi-head attention. It allows the model to "attend" to relative spatial positions.
- **Query/Key/Value**: Enriched with spatial embeddings.
- **Spatial Bias**: A dedicated bias term added to the attention scores based on the relative distance between tokens $(x_i - x_j, y_i - y_j)$.

### 3. GCN Refinement
After the Transformer layers, the sequence of hidden states is treated as a graph.
- A **GCN** layer aggregates information from neighbors defined by the spatial graph.
- This helps resolving ambiguities where semantic context (text) is insufficient, but spatial layout (columns, tables) is clear.

## Extending the Project

### Adding a New Dataset
1. Create a loader in `data/dataset.py` that returns a dictionary with `words`, `bboxes`, and `ner_tags`.
2. Ensure bounding boxes are normalized to 0-1000.

### Using a Different Backbone
Modify `tokenization/text_embedding.py` to use `bert-base-uncased`, `roberta-base`, or `microsoft/layoutlm-base-uncased`.

### Visual Backbone
To add image features (e.g., ResNet):
1. Add an image encoder in `model/formnet.py`.
2. Fuse image features with text+spatial embeddings in `CombinedEmbedding`.
