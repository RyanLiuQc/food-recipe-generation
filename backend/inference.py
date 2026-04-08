import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from lstm_model import LSTMGenerator

# GPT-2's tokenizer is better for generation tasks than BERT's
tokenizer = AutoTokenizer.from_pretrained("gpt2")
# GPT-2 doesn't have a default padding token, so we set it to the EOS token
tokenizer.pad_token = tokenizer.eos_token


# 1. Recreate the empty model architecture (must match exact dimensions used in training)
loaded_model = LSTMGenerator(
    vocab_size=tokenizer.vocab_size,  # 50257
    embedding_dim=256,
    hidden_dim=512
)
# 3. Move to GPU and set to Evaluation Mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 2. Load the saved weights
#loaded_model.load_state_dict(torch.load('recipe_generator_lstm.pth'))
loaded_model.load_state_dict(
    torch.load('recipe_generator_lstm.pth', map_location=device, weights_only=True)
)


loaded_model.to(device)
loaded_model.eval() # tells pytorch we are not training

def generate_recipe(ingredients_list, recipe_title="My Custom Recipe", max_length=300, temperature=0.8):
    """
    Takes ingredients, formats them using our training landmarks, and generates the instructions.
    Temperature controls creativity:
        Lower (e.g., 0.3) = safer, more repetitive.
        Higher (e.g., 1.2) = creative, potentially chaotic.
    """
    model = loaded_model
    model.eval()
    device = next(model.parameters()).device

    # 1. Format the "Seed" text exactly how the model saw it during training
    ingredients_str = "\n".join([f"• {ing.strip()}" for ing in ingredients_list])
    seed_text = f"📗 {recipe_title}\n🥕\n{ingredients_str}\n📝\n"

    # 2. Tokenize the seed text
    input_ids = tokenizer.encode(seed_text, return_tensors='pt').to(device)
    generated_sequence = input_ids[0].tolist()

    print("--- Generating Recipe ---")

    with torch.no_grad():
        for _ in range(max_length):
            # We must pass the sequence into the model.
            # To prevent memory errors, we only pass the last 300 tokens (our MAX_SEQ_LENGTH from training)
            seq_input = torch.tensor([generated_sequence[-300:]]).to(device)

            # Re-initialize hidden state for this forward pass
            h = model.init_hidden(1, device)

            # Get predictions
            logits, _ = model(seq_input, h)

            # We only care about the model's prediction for the *very last* token
            next_token_logits = logits[0, -1, :]

            # 3. Apply Temperature Scaling
            next_token_logits = next_token_logits / temperature

            # 4. Convert logits to probabilities using Softmax
            probs = F.softmax(next_token_logits, dim=-1)

            # 5. Sample the next token based on those probabilities
            next_token = torch.multinomial(probs, num_samples=1).item()

            # Append to our sequence
            generated_sequence.append(next_token)

            # Stop generating if the model outputs the padding/end token
            if next_token == tokenizer.eos_token_id:
                break

    # 6. Decode the final sequence back into readable text
    return tokenizer.decode(generated_sequence)
    