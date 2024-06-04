from transformers import GPT2Model
from transformers import GPT2TokenizerFast
from copy import deepcopy
import torch
import torch.nn as nn

def only_bias_training(model: GPT2Model):
    for name, param in model.named_parameters():
        if 'bias' in name:
            continue
        param.requires_grad = False

def freeze_first_k_transformer_layers(k: int, model: GPT2Model):
    assert 0 <= k <= len(model.layers) # Let's not use this for full finetuning
    for i in range(k):
        for param in model.layers[i].parameters():
            param.requires_grad = False

def freeze_last_k_transformer_layers(k: int, model: GPT2Model):
    n = len(model.layers)

    assert 0 <= k <= n
    for i in range(n-k, n):
        for param in model.layers[i].parameters():
            param.requires_grad = False

def only_kth_transformer_layer(k: int, model: GPT2Model):
    n = len(model.layers)
    assert 0 <= k <= n
    for i in range(0, n):
        if i == k:
            continue

        for param in model.layers[i].parameters():
            param.requires_grad = False

def check_two_gpt2_models_params(model1: GPT2Model, model2: GPT2Model):
    assert len(model1.layers) == len(model2.layers)
    for i in range(len(model1.layers)):
        param1 = list(model1.layers[i].parameters())[0]
        param2 = list(model2.layers[i].parameters())[0]
        if torch.all(param1.eq(param2)):
            print(f"===Layer {i} is the same")
        else:
            print(f"===Layer {i} is different")


if __name__ == "__main__":
    if torch.backends.mps:
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = GPT2Model.from_pretrained('gpt2').to(device)

    loss_fn = nn.MSELoss()
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    # Example data
    batch_seqs = ["Hello world!", "My name is"]
    batch_x = tokenizer(batch_seqs, return_tensors="pt").to(device=device)

    def train_for_10_iters(model, optimizer, loss_fn, inputs):
        for _ in range(10):
            # Get the per-datapoint representation
            outputs = model.forward(**batch_x)['last_hidden_state']
            outputs = torch.mean(outputs, dim=1)
            loss = loss_fn(outputs, torch.zeros_like(outputs))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    copied_model = deepcopy(model)
    optimizer = torch.optim.Adam(copied_model.parameters(), lr=0.01)
    freeze_first_k_transformer_layers(6, copied_model)
    train_for_10_iters(copied_model, optimizer, loss_fn, batch_x)

    check_two_gpt2_models_params(model, copied_model)
