# Text Generation using RNN and LSTM

A deep learning project to generate poetry using Recurrent Neural Networks (RNN) and LSTM models trained on a dataset of 100 English poems.

## Features

- **Text Preprocessing**: Tokenized the dataset into words and built word-to-index mappings.
- **Model Architectures**:
  - **OneHotRNN**: Uses one-hot encoded input sequences.
  - **EmbeddingLSTM**: Uses a trainable embedding layer for word representation.
- **Loss & Optimization**: Cross-entropy loss with Adam optimizer.
- **Training**: Compared both models over 100 epochs for performance and loss.
- **Poem Generation**: Generates text continuation from seed words.

## Installation

Clone the repository:
```bash
git clone https://github.com/AyushHL/Poem-Text-Generation.git
cd Poem-Text-Generation
```

Install dependencies:
```bash
pip install torch
```

Ensure your dataset is placed at:
```
/kaggle/input/poems-dataset/poems-100.csv
```

Run the training script:
```bash
python main.py
```

## Usage

Generate poems using a seed line:
```python
seed_text = "I wandered lonely as a"
print(generate_poem(model, seed_text))
```

## Results

| Model         | Final Loss | Training Time |
|---------------|------------|----------------|
| OneHotRNN     | 0.0108     | ~1674 seconds  |
| EmbeddingLSTM | 0.0003     | ~878 seconds   |

**Example Output**

- **Input**: `"I wandered lonely as a"`

- **OneHotRNN Output**:  
  `"... desert the child that by the women, sings of the country fast..."`

- **EmbeddingLSTM Output**:  
  `"... this all that is not by the great tomb of man. The golden sun..."`

## Tech Stack

- Python
- PyTorch
- RNN / LSTM
- NLP (Tokenization, Embedding)
- CSV File I/O

## Future Improvements

- Add attention mechanisms or transformer layers.
- Hyperparameter tuning for better generation quality.

## License

This project is open source and available under the [MIT License](LICENSE).
