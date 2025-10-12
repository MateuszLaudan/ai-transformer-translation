#!/usr/bin/env python
# coding: utf-8

# <!-- # Summarization model -->

# In[ ]:


import torch
import torch.nn as nn
import json

seed = 42
torch.manual_seed(seed)  # PyTorch CPU
torch.cuda.manual_seed(seed)  # PyTorch GPU
torch.cuda.manual_seed_all(seed)  # Multi-GPU

# Check if GPU is available
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = 'cpu'
print(f"Using device: {device}")

import os


# # Load dataset

# In[ ]:


from datasets import load_dataset

ds = load_dataset("FrancophonIA/french-to-english", split="train", streaming=True)
print(ds)


# # Get Tokenizer - BartTokenizer with 50265 vocab size

# In[ ]:


from transformers import BartTokenizer

tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
VOCAB_SIZE = tokenizer.vocab_size

SRC_MAX_SEQ = 50
TGT_MAX_SEQ = 50
max_examples = 600000


# # Prepare data - save data

# In[ ]:


from transformers import pipeline

model_ckpt = "papluca/xlm-roberta-base-language-detection"
pipe = pipeline("text-classification", model=model_ckpt, device=device)


# In[ ]:


# inputs = []
# outputs = []
# i = 0

# long_samples = 0
# wrong_lang = 0

# for j, data in enumerate(ds):
#     src_sentence = tokenizer.encode(str(data['en']))
#     tgt_sentence = tokenizer.encode(str(data['fr']))

#     if len(src_sentence) <= SRC_MAX_SEQ and len(tgt_sentence) <= TGT_MAX_SEQ:
#         try:
#             if not data.get('fr') or not isinstance(data['fr'], str):
#                 continue  # Skip invalid entries
#             else:
#                 res_fr = pipe(data['fr'], top_k=1, truncation=True)[0]['label']
#                 res_en = pipe(data['en'], top_k=1, truncation=True)[0]['label']

#                 if res_fr == 'fr' and res_en == 'en':       
#                     inputs.append(src_sentence)
#                     outputs.append(tgt_sentence)
#                     i += 1
#                 else:
#                     wrong_lang += 1
#         except Exception as e:
#             print(f"Error {e}")
#             continue       
#     else:
#         long_samples += 1
#     if i == max_examples:
#         break

# # Stats
# print('Number of loops until max_examples: ', j + 1)
# print('Final dataset size: ', len(inputs))
# print('Long samples (>50 tokens each): ', long_samples)
# print('Wrongly detected translations: ', wrong_lang)
# # Save as JSON
# with open('tokenized_data.json', 'w') as f:
#     json.dump({'inputs': inputs, 'outputs': outputs}, f)


# # Load data

# In[ ]:


inputs = []
outputs = []

with open('tokenized_data.json', 'r') as f:
    data = json.load(f)
    inputs = data['inputs'][:max_examples]
    outputs = data['outputs'][:max_examples]
print(len(inputs), len(outputs))


# # Clean data

# In[ ]:


"""To set determined len for src and tgt --> also with pad tokens"""


def clean_data(sentences, max_seq_length):
    for i, sentence in enumerate(sentences):
        if len(sentence) < max_seq_length:
            sentences[i] = sentence + [tokenizer.pad_token_id] * (max_seq_length - len(sentence))


clean_data(inputs, SRC_MAX_SEQ)
clean_data(outputs, TGT_MAX_SEQ)


# In[ ]:


for step, (inp, out) in enumerate(zip(inputs, outputs)):
    print(tokenizer.decode(inp, skip_special_tokens=True), '######', tokenizer.decode(out, skip_special_tokens=True))
    if step == 10:
        break


# In[ ]:


inputs = torch.tensor(inputs)
targets = torch.tensor(outputs)
assert len(inputs) == len(targets), "Number of articles and summaries must be the same"
print('Max src seq len:', SRC_MAX_SEQ)
print('Max tgt seq len:', TGT_MAX_SEQ)
print('Number of examples:', len(inputs))
print("Number of vocab size:", VOCAB_SIZE)


# # Custom Dataset and DataLoader with train and val data (80%, 20%)

# In[ ]:


from torch.utils.data import random_split, Dataset


class CustomDataset(Dataset):
    def __init__(self, inputs: torch.Tensor, outputs: torch.Tensor):
        self.inputs = inputs
        self.outputs = outputs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]


dataset = CustomDataset(inputs, targets)
train_data, val_data = random_split(dataset, [0.8, 0.2])
print(f"Number of training samples: {len(train_data)}")
print(f"Number of validation samples: {len(val_data)}")


# In[ ]:


from torch.utils.data import DataLoader

BATCH_SIZE = 128  # TODO: expand batch size for transformer architecture

train_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    # drop_last=True
)

val_loader = DataLoader(
    val_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    # drop_last=True
)

print(f"Number of batches in training set: {len(train_loader)}")
print(f"Number of batches in validation set: {len(val_loader)}")


# In[ ]:


import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads 

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # TODO: use F..scaled_dot_product_attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        attn_probs = torch.softmax(attn_scores, dim=-1)

        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        batch_size, seq_length, _ = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_length, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output


# In[ ]:


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        # Assumption: GELU outperforms ReLU which leads to 'ReLU dead neuron problem': https://arxiv.org/pdf/1606.08415
        self.func = nn.GELU(approximate='tanh')
        # self.func = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.func(self.fc1(x)))


# In[ ]:


"""NOTE: There is no evidence that positional encoding is better than simple learnable embeddings."""

#
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_seq_length):
#         super(PositionalEncoding, self).__init__()
#
#         pe = torch.zeros(max_seq_length, d_model)
#         position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
#
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#
#         self.register_buffer('pe', pe.unsqueeze(0))
#
#     def forward(self, x):
#         return x + self.pe[:, :x.size(1)]


# In[ ]:


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)  # TODO: one dropout layer or two?

    def forward(self, x, mask):
        # Normalization before sub-blocks: as described at: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
        residual_x = self.norm1(x)  # residual_x described at: https://arxiv.org/pdf/1904.10509
        attn_output = self.self_attn(residual_x, residual_x, residual_x, mask)
        x = x + self.dropout(attn_output)
        residual_x = self.norm2(x)  # Normalization before sub-blocks
        ff_output = self.feed_forward(residual_x)
        x = x + self.dropout(ff_output)
        return x


# In[ ]:


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        # Normalization before sub-blocks: as described at: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
        residual_x = self.norm1(x)
        attn_output = self.self_attn(residual_x, residual_x, residual_x, tgt_mask)
        x = x + self.dropout(attn_output)
        residual_x = self.norm2(x)  # Normalization before sub-blocks
        attn_output = self.cross_attn(residual_x, enc_output, enc_output, src_mask)
        x = x + self.dropout(attn_output)
        residual_x = self.norm3(x)  # Normalization before sub-blocks
        ff_output = self.feed_forward(residual_x)
        x = x + self.dropout(ff_output)
        return x


# In[ ]:


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_src_seq_len,
                 max_tgt_seq_len, dropout):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        # self.src_positional_encoding = PositionalEncoding(d_model, max_src_seq_len)
        # self.tgt_positional_encoding = PositionalEncoding(d_model, max_tgt_seq_len)
        self.src_positional_encoding = nn.Embedding(max_src_seq_len,
                                                    d_model)  # instead of cos and sin functions (in PositionalEncoding)
        self.tgt_positional_encoding = nn.Embedding(max_tgt_seq_len, d_model)

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        # One more LayerNorm as described: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
        self.ln_f = nn.LayerNorm(d_model)
        self.output_layer = nn.Linear(d_model, tgt_vocab_size,
                                      bias=False)  # bias=False as depicted: https://github.com/karpathy/nanoGPT/blob/master/model.py
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        src_mask = (src != tokenizer.pad_token_id).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != tokenizer.pad_token_id).unsqueeze(1).unsqueeze(3)
        src_mask, tgt_mask = src_mask.to(device), tgt_mask.to(device)
        seq_length = tgt.size(1)
        casual_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        casual_mask = casual_mask.to(device)
        tgt_mask = tgt_mask & casual_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        # src_embedded = self.dropout(self.src_positional_encoding(self.encoder_embedding(src)))
        # tgt_embedded = self.dropout(self.tgt_positional_encoding(self.decoder_embedding(tgt)))

        src_pos = torch.arange(0, src.size(1)).to(device)  # [0, 1, 2 ... src.size(1)]
        src_pos = self.src_positional_encoding(src_pos)
        src_embedded = self.encoder_embedding(src)
        src_embedded = self.dropout(src_pos + src_embedded)

        tgt_pos = torch.arange(0, tgt.size(1)).to(device)
        tgt_pos = self.tgt_positional_encoding(tgt_pos)
        tgt_embedded = self.decoder_embedding(tgt)
        tgt_embedded = self.dropout(tgt_pos + tgt_embedded)

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        # One more LayerNorm as described: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
        dec_output = self.ln_f(dec_output)
        output = self.output_layer(dec_output)
        return output


# In[ ]:


src_vocab_size = VOCAB_SIZE
tgt_vocab_size = VOCAB_SIZE
d_model = 128  # TODO: experiment with model complexity - it can lead to overfitting
num_heads = 1  # TODO: layers + heads have impact on final results: https://medium.com/@ccibeekeoc42/unveiling-the-transformer-impact-of-layers-and-attention-heads-in-audio-classification-58747d52b794
num_layers = 6  # TODO: how many layers? as above, model complexity; smaller BART uses 6: https://arxiv.org/pdf/1910.13461
d_ff = d_model * 4
max_src_seq_len = SRC_MAX_SEQ
max_tgt_seq_len = TGT_MAX_SEQ
dropout = 0.1


# In[ ]:


transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_src_seq_len,
                          max_tgt_seq_len,
                          dropout)
transformer.to(device)
num_parameters = sum(p.numel() for p in transformer.parameters())
print(f"Number of parameters: {num_parameters/1000000} M")


# # Train model

# In[ ]:


import torch
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup
from torch import optim
from tqdm import tqdm

torch.manual_seed(seed) 
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

lr = 0.0007
optimizer = optim.Adam(
    transformer.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.00
)

# TODO: experiment with gradient accumulation
num_epochs = 150
# TODO: add warmup steps
num_training_steps = len(train_loader) * num_epochs
num_warmup_steps = int(0.6 * num_training_steps) # TODO: experiment with that

scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
)

# Tracking history
lr_history = []
loss_history = []
val_loss_history = []

for epoch in range(num_epochs):
    epoch_loss = 0.0
    transformer.train()

    # Progress bar for training
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

    for step, (src_data, tgt_data) in enumerate(progress_bar):
        src_data, tgt_data = src_data.to(device), tgt_data.to(device)

        # Forward pass
        output = transformer(src_data, tgt_data[:, :-1])

        # Compute loss
        loss = F.cross_entropy(
            output.contiguous().view(-1, tgt_vocab_size),
            tgt_data[:, 1:].contiguous().view(-1),
            ignore_index=tokenizer.pad_token_id
        )

        # Track loss
        loss_history.append(loss.item())
        epoch_loss += loss.item()
        lr_history.append(optimizer.param_groups[0]["lr"])

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        progress_bar.set_postfix({"Batch Loss": loss.item()})

    avg_epoch_loss = epoch_loss / len(train_loader)
    print(f"Epoch: {epoch + 1}, Loss: {avg_epoch_loss:.4f}")

    transformer.eval()
    total_val_loss = 0.0

    with torch.no_grad():
        for val_src_data, val_tgt_data in val_loader:
            val_src_data, val_tgt_data = val_src_data.to(device), val_tgt_data.to(device)

            # Forward pass
            val_output = transformer(val_src_data, val_tgt_data[:, :-1])

            val_loss = F.cross_entropy(
                val_output.contiguous().view(-1, tgt_vocab_size),
                val_tgt_data[:, 1:].contiguous().view(-1),
                ignore_index=tokenizer.pad_token_id
            )

            total_val_loss += val_loss.item()

    # Calculate average validation loss
    avg_val_loss = total_val_loss / len(val_loader)
    val_loss_history.append(avg_val_loss)
    print(f"Average Validation Loss: {avg_val_loss:.4f}")


# # Show charts with lr and loss

# In[ ]:


from matplotlib import pyplot as plt

assert len(lr_history) == len(
    loss_history
), "Length of lr_history and loss_history must be the same"

fig, ax1 = plt.subplots()

ax1.set_title("Learning Rate vs. Loss")
ax1.set_xlabel("Training Step")
ax1.set_ylabel("Loss", color="tab:red")
ax1.plot(range(len(loss_history)), loss_history, color="tab:red", label="Training Loss")
ax1.tick_params(axis="y", labelcolor="tab:red")

epoch_interval = len(loss_history) // len(val_loss_history)
val_x = [
    i * epoch_interval for i in range(len(val_loss_history))
]
ax1.plot(
    val_x,
    val_loss_history,
    color="tab:orange",
    marker="o",
    linestyle="dashed",
    label="Validation Loss",
)

ax2 = ax1.twinx()
ax2.set_ylabel("Learning Rate", color="tab:blue")
ax2.plot(
    range(len(lr_history)),
    lr_history,
    color="tab:blue",
    linestyle="--",
    label="Learning Rate",
)
ax2.tick_params(axis="y", labelcolor="tab:blue")

# Legends
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")

plt.show()


# In[ ]:


PATH = f"model_{d_model}.pt"
PATH


# In[ ]:


# torch.save(transformer.state_dict(), PATH)


# In[ ]:


transformer_output = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_src_seq_len,
                          max_tgt_seq_len,
                          dropout)
transformer_output.to(device)
transformer_output.load_state_dict(torch.load(PATH, weights_only=True))


# # Model inference

# In[ ]:


import evaluate
import httpx
from dotenv import load_dotenv
import os
import statistics

load_dotenv()

deepl_auth_key = os.getenv("DEEPL_AUTH_KEY")

# -----------------------------
# Metrics setup
# -----------------------------
rouge = evaluate.load("rouge")
sacrebleu = evaluate.load("sacrebleu")
meteor = evaluate.load("meteor")
bleu = evaluate.load("bleu")

# -----------------------------
# Inference function
# -----------------------------
def inference(input_text: str, tokenizer, model, max_length: int = TGT_MAX_SEQ, device='cuda'):
    """
    Perform auto-regressive translation inference.
    """
    model.eval()
    tokens = tokenizer.encode(input_text)
    src_data = torch.tensor(tokens).unsqueeze(0).to(device)
    tgt_data = torch.tensor([tokenizer.bos_token_id]).unsqueeze(0).to(device)

    with torch.no_grad():
        for _ in range(max_length):
            output = model(src_data, tgt_data)
            next_token_id = output[:, -1, :].argmax(dim=-1).item()
            tgt_data = torch.cat([tgt_data, torch.tensor([[next_token_id]]).to(device)], dim=1)
            if next_token_id == tokenizer.eos_token_id:
                break

    translated_tokens = tgt_data.squeeze().tolist()
    return tokenizer.decode(translated_tokens, skip_special_tokens=True)

# -----------------------------
# Dataset example translation
# -----------------------------
def translate_dataset_example(model, tokenizer, dataset_loader, index=0):
    """
    Translate a single example from a given dataset loader.
    Returns src sentence, model translation, and reference translation.
    """
    dataset = dataset_loader.dataset
    src_input = tokenizer.decode(dataset[index][0].tolist(), skip_special_tokens=True)
    reference = tokenizer.decode(dataset[index][1].tolist(), skip_special_tokens=True)
    translation = inference(src_input, tokenizer, model)

    return src_input, translation, reference

# -----------------------------
# Compute metrics for one pair
# -----------------------------
def compute_metrics(prediction: str, reference: str):
    """
    Compute all relevant metrics between a prediction and reference.
    Returns a dictionary.
    """
    preds = [prediction]
    refs = [[reference]]

    results = {
        'ROUGE-1': rouge.compute(predictions=preds, references=refs)['rouge1'],
        'ROUGE-2': rouge.compute(predictions=preds, references=refs)['rouge2'],
        'ROUGE-L': rouge.compute(predictions=preds, references=refs)['rougeL'],
        'SacreBLEU': sacrebleu.compute(predictions=preds, references=refs)['score'],
        'METEOR': meteor.compute(predictions=preds, references=refs)['meteor'],
        'BLEU': bleu.compute(predictions=preds, references=refs, smooth=True)['bleu']
    }
    return results

# -----------------------------
# Translate using DeepL
# -----------------------------
def translate_deepl(src_text: str, target_lang='FR'):
    """
    Send text to DeepL API and return the translation.
    """
    json_data = {"text": [src_text], "target_lang": target_lang}
    headers = {"Authorization": f"DeepL-Auth-Key {deepl_auth_key}"}
    response = httpx.post("https://api-free.deepl.com/v2/translate", json=json_data, headers=headers)
    response.raise_for_status()
    return response.json()["translations"][0]["text"]

# -----------------------------
# Evaluate multiple examples
# -----------------------------
def evaluate_model(model, tokenizer, dataset_loader, num_examples):
    """
    Evaluate model on dataset and compare with DeepL.
    Returns aggregated metrics (mean and std).
    """
    metrics_names = ['ROUGE-1','ROUGE-2','ROUGE-L','SacreBLEU','METEOR','BLEU']

    local_metrics_all = {k: [] for k in metrics_names}
    deepl_metrics_all = {k: [] for k in metrics_names}
    deepl_metrics_to_local_all = {k: [] for k in metrics_names}

    for i in range(num_examples):
        src, pred, ref = translate_dataset_example(model, tokenizer, dataset_loader, index=i)
        deepl_pred = translate_deepl(src)

        local_metrics = compute_metrics(pred, ref)
        deepl_metrics = compute_metrics(deepl_pred, ref)
        deepl_metrics_to_local = compute_metrics(pred, deepl_pred)

        for k in metrics_names:
            local_metrics_all[k].append(local_metrics[k])
            deepl_metrics_all[k].append(deepl_metrics[k])
            deepl_metrics_to_local_all[k].append(deepl_metrics_to_local[k])

        print(f"Example {i+1}:")
        print("SRC:       ", src)
        print("TRANSLATION:      ", pred)
        print("REFERENCE: ", ref)
        print("DEEPL_PRED:     ", deepl_pred)
        print("Local Metrics:", local_metrics)
        print("DeepL Metrics:", deepl_metrics)
        print("DeepL Metrics comparing to Local Model", deepl_metrics_to_local)
        print('-'*80)

    def summarize(metrics_dict):
        return {k: (statistics.mean(v), statistics.stdev(v)) for k, v in metrics_dict.items()}

    local_summary = summarize(local_metrics_all)
    deepl_summary = summarize(deepl_metrics_all)
    deepl_to_local_summary = summarize(deepl_metrics_to_local_all)

    print(f"\n===== RESULTS after {num_examples} samples =====")
    print("LOCAL MODEL (mean ± std):", local_summary)
    print("DEEPL (mean ± std):", deepl_summary)
    print("DEEPL vs Local (mean ± std):", deepl_to_local_summary)

# -----------------------------
# Usage
# -----------------------------
evaluate_model(
    transformer_output, tokenizer, val_loader, num_examples=10
)

