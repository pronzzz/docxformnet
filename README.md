# FormNet: Document Extraction with Rich Attention and GCN

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-ee4c2c)
![License](https://img.shields.io/badge/license-MIT-green)

**FormNet** is a state-of-the-art document extraction system designed to parse complex forms with irregular layouts, tables, and columns. It significantly improves upon standard sequence models by leveraging **Rich Attention** for 2D spatial reasoning and **Graph Convolutional Networks (GCN)** for structural context.

## 🚀 Key Features

- **Spatial Embedding**: Learns rich representations from 2D token coordinates.
- **Rich Attention**: A modified self-attention mechanism that captures direct spatial relationships between tokens.
- **GCN Super-Tokens**: Refines token representations using a spatial graph (K-NN) to aggregate neighborhood context.
- **Robust Serialization**: Minimizes layout-serialization errors common in flat BERT-like models.

## 🏗 Architecture Overview

The system is built with modular components:

1.  **Embedding Layer**: Fuses DistilBERT text embeddings with learnable spatial embeddings.
2.  **Rich Attention**: Incorporates 2D spatial biases into the self-attention mechanism, allowing the model to "see" the layout.
3.  **GCN Module**: Refines token representations using a K-Nearest Neighbors (KNN) graph constructed from bounding box coordinates.
4.  **Decoder**: A classification head for BIO tagging (Entity Extraction) or sequence generation.

## 📂 Project Structure

For a detailed explanation of the codebase, see the [Developer Guide](GUIDE.md).

## 🛠 Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/pronzzz/docxformnet.git
    cd docxformnet
    ```

2.  Install dependencies:
    ```bash
    pip3 install -r requirements.txt
    ```

## 📊 Dataset Preparation

FormNet supports CORD (Receipts) and FUNSD (Forms) datasets.

```bash
python3 data/download_datasets.py
```
*The script automatically handles dataset downloading and formatting using the Hugging Face `datasets` library.*

## 🏋️‍♀️ Training

Train the model on the CORD dataset:

```bash
python3 train.py --data_dir data/CORD --epochs 10 --batch_size 4
```

This command will:
- Initialize the dataset and tokenizer.
- constructing spatial graphs on-the-fly during batch collation.
- Train the FormNet model and save checkpoints to `checkpoints/`.

## 📈 Evaluation

Evaluate a trained model on the test set:

```bash
python3 evaluate.py --checkpoint_path checkpoints/model_epoch_10.pt --data_dir data/CORD
```

## 🔍 Visualization

You can visualize the model's understanding of the document layout (bounding boxes and labels):

```python
from utils.visualize import visualize_document
# ... load your data ...
visualize_document("path/to/image.png", words, bboxes, labels)
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
