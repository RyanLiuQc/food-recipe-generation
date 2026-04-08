# Food Recipe Generation

A Machine Learning application that generates a recipe. By providing a list of available ingredients, this autoregressive language model generates a structured, step-by-step cooking recipe, formatted ingredient list, and sequential instructions.

This project was developed as part of the MAIS 202 (McGill Artificial Intelligence Society) Machine Learning Bootcamp.

## Dataset

https://www.kaggle.com/datasets/sarthak71/food-recipes

## Features

* **Autoregressive Model:** Built a Long Short-Term Memory (LSTM) network from scratch using PyTorch.
* **Tokenization:** Utilizes the GPT-2 Byte-Pair Encoding (BPE) tokenizer to effectively handle complex culinary vocabulary and measurements.
* **Recipe format:** Learns and reproduces specific visual landmarks to cleanly format recipes (📗 for Title, 🥕 for Ingredients, 📝 for Instructions).
* **Web App:** Features a React frontend and a FastAPI backend for real-time model inference.

## How it Works

1. **Data Formatting:** The model was trained on a dataset of 8,000+ recipes from Kaggle. The raw CSV data was transformed into unified strings using visual structural landmarks. This taught the model the anatomy of a recipe. 
2. **Training:** Sequences of up to 300 tokens were fed into a 2-layer LSTM (hidden dimension: 512). The model acts as a next-token predictor, trained using `CrossEntropyLoss` and gradient clipping to prevent exploding gradients.
3. **Inference:** The user's input ingredients are formatted into a "seed prompt". The model predicts the next token, applies temperature scaling to the softmax probabilities, samples the token, and autoregressively feeds it back into the loop until an End-of-Sequence token is generated.

## Tech Stack

**Machine Learning & Data Processing:**
* Python
* PyTorch (Custom LSTM Architecture)
* Hugging Face `transformers` (GPT-2 Tokenizer)
* Pandas & NumPy (Data processing)
* Jupyter Notebook (Data exploration & training pipeline)

**Backend:**
* FastAPI
* Uvicorn

**Frontend:**
* React.js
* Node.js

## Getting Started

### Prerequisites
* Python 3.10
* Node.js & npm
* Conda (recommended for managing Python environments)

### 1. Clone the Repository
```bash
git clone [https://github.com/RyanLiuQc/food-recipe-generation.git](https://github.com/RyanLiuQc/food-recipe-generation.git)
cd food-recipe-generation
```
### 2. Install dependencies 

backend
```bash
# using conda
conda env create -f environment.yml
conda activate lstm_env

# otherwise use pip
pip install fastapi uvicorn pydantic transformers torch

# Start the ASGI server
cd backend
uvicorn main:app --reload

```

frontend
```bash
cd frontend
npm install
npm start
```