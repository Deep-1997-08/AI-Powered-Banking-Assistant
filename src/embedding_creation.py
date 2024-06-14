from transformers import MPNetTokenizer, MPNetModel
import torch

def create_embeddings(texts):
    tokenizer = MPNetTokenizer.from_pretrained("microsoft/mpnet-base")
    model = MPNetModel.from_pretrained("microsoft/mpnet-base")
    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).detach().numpy())
    return embeddings
