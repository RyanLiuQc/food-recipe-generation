import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from lstm_model import LSTMGenerator
import re

# GPT-2's tokenizer is better for generation tasks than BERT's
tokenizer = AutoTokenizer.from_pretrained("gpt2")
# GPT-2 doesn't have a default padding token, so we set it to the EOS token
tokenizer.pad_token = tokenizer.eos_token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loaded_model = LSTMGenerator(vocab_size=tokenizer.vocab_size, embedding_dim=256, hidden_dim=512)

loaded_model.load_state_dict(
    torch.load('recipe_generator_0408.pth', map_location=device)
)

# 3. Move to GPU and set to Evaluation Mode
loaded_model.to(device)
loaded_model.eval() 
# inference.py

def clean_output(text: str) -> str:
    # Remove tokens like <END>, <START>, etc.
    text = re.sub(r"<[^>]+>", "", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()

def generate_recipe(ingredients_list, recipe_title="Custom Recipe",
                    max_length=300, temperature=0.8, top_k=50):

    model = loaded_model
    tok = tokenizer

    model.eval()
    device = next(model.parameters()).device

    ingredients_str = "\n".join([ing.strip() for ing in ingredients_list])

    seed_text = (
        f"TITLE: {recipe_title}\n"
        f"INGREDIENTS:\n{ingredients_str}\n"
        f"INSTRUCTIONS:\n"
    )

    input_ids = tok.encode(seed_text, return_tensors='pt').to(device)
    generated = input_ids

    h = model.init_hidden(1, device)

    with torch.no_grad():
        for _ in range(max_length):
            outputs, h = model(generated[:, -300:], h)
            logits = outputs[:, -1, :] / temperature

            topk_vals, topk_idx = torch.topk(logits, top_k)
            probs = torch.softmax(topk_vals, dim=-1)

            sampled_idx = torch.multinomial(probs, 1)
            next_token = topk_idx.gather(-1, sampled_idx)

            generated = torch.cat((generated, next_token), dim=1)

            if next_token.item() == tok.eos_token_id:
                break

    return clean_output(tok.decode(generated[0]))