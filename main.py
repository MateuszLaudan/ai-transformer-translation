# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# <!-- # Summarization model -->

# %%
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

# import os


# %% [markdown]
# # Load dataset

# %%

from datasets import load_dataset

ds = load_dataset("FrancophonIA/french-to-english", split="train", streaming=True)
print(ds)

# %% [markdown]
# # Get Tokenizer - BartTokenizer with 50265 vocab size

# %%
from transformers import BartTokenizer

tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
VOCAB_SIZE = tokenizer.vocab_size

SRC_MAX_SEQ = 50
TGT_MAX_SEQ = 50
max_examples = 600000

# %% [markdown]
# # Prepare data - save data

# %%
from transformers import pipeline

model_ckpt = "papluca/xlm-roberta-base-language-detection"
pipe = pipeline("text-classification", model=model_ckpt, device=device)

# %%

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

# %% [markdown]
# # Load data

# %%
inputs = []
outputs = []

with open('tokenized_data.json', 'r') as f:
    data = json.load(f)
    inputs = data['inputs'][:max_examples]
    outputs = data['outputs'][:max_examples]
print(len(inputs), len(outputs))
# for inp, out in zip(inputs, outputs):
#     out = tokenizer.decode(out, skip_special_tokens=True)
#     inp = tokenizer.decode(inp, skip_special_tokens=True)
#     print(inp)
#     print(out)
#     print()
#     res = pipe(out, top_k=1, truncation=True)[0]['label']
#     if res != 'fr':
#         print(res, out)

# %% [markdown]
# # Clean data

# %%
"""To set determined len for src and tgt --> also with pad tokens"""


def clean_data(sentences, max_seq_length):
    for i, sentence in enumerate(sentences):
        if len(sentence) < max_seq_length:
            sentences[i] = sentence + [tokenizer.pad_token_id] * (max_seq_length - len(sentence))


clean_data(inputs, SRC_MAX_SEQ)
clean_data(outputs, TGT_MAX_SEQ)

# %%
for step, (inp, out) in enumerate(zip(inputs, outputs)):
    print(tokenizer.decode(inp, skip_special_tokens=True), '######', tokenizer.decode(out, skip_special_tokens=True))
    if step == 10:
        break

# %%
inputs = torch.tensor(inputs)
targets = torch.tensor(outputs)
assert len(inputs) == len(targets), "Number of articles and summaries must be the same"
print('Max src seq len:', SRC_MAX_SEQ)
print('Max tgt seq len:', TGT_MAX_SEQ)
print('Number of examples:', len(inputs))
print("Number of vocab size:", VOCAB_SIZE)

# %% [markdown]
# # Custom Dataset and DataLoader with train and val data (80%, 20%)

# %%
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

# %%
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


# %%
print(tokenizer.decode(train_loader.dataset[0][0], skip_special_tokens=True) )
print(tokenizer.decode(train_loader.dataset[0][1], skip_special_tokens=True) )

# %%
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads 

        self.W_q = nn.Linear(d_model, d_model)  # Query transformation
        self.W_k = nn.Linear(d_model, d_model)  # Key transformation
        self.W_v = nn.Linear(d_model, d_model)  # Value transformation
        self.W_o = nn.Linear(d_model, d_model)  # Output transformation

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # TODO: use F..scaled_dot_product_attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        attn_probs = torch.softmax(attn_scores, dim=-1)

        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output


# %%
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        # Assumption: GELU outperforms ReLU which leads to 'ReLU dead neuron problem': https://arxiv.org/pdf/1606.08415
        self.gelu = nn.GELU(approximate='tanh')

    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))


# %%
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

# %%
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


# %%
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

# %%



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


# %%
src_vocab_size = VOCAB_SIZE
tgt_vocab_size = VOCAB_SIZE
d_model = 128  # TODO: experiment with model complexity - it can lead to overfitting
num_heads = 1  # TODO: layers + heads have impact on final results: https://medium.com/@ccibeekeoc42/unveiling-the-transformer-impact-of-layers-and-attention-heads-in-audio-classification-58747d52b794
num_layers = 6  # TODO: how many layers? as above, model complexity; smaller BART uses 6: https://arxiv.org/pdf/1910.13461
d_ff = d_model * 4
max_src_seq_len = SRC_MAX_SEQ
max_tgt_seq_len = TGT_MAX_SEQ
dropout = 0.1

# %%
transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_src_seq_len,
                          max_tgt_seq_len,
                          dropout)
transformer.to(device)
num_parameters = sum(p.numel() for p in transformer.parameters())
print(f"Number of parameters: {num_parameters/1000000} M")


# %% [markdown]
# # Train model

# %%
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
num_warmup_steps = int(0.1 * num_training_steps) # TODO: experiment with that
 
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


# %%
# import gc
# del transformer
# gc.collect()

# %% [markdown]
# # Show charts with lr and loss

# %%
from matplotlib import pyplot as plt

# Ensure all histories are properly formatted
assert len(lr_history) == len(
    loss_history
), "Length of lr_history and loss_history must be the same"

# Create figure and primary y-axis for Loss
fig, ax1 = plt.subplots()

ax1.set_title("Learning Rate vs. Loss")
ax1.set_xlabel("Training Step")
ax1.set_ylabel("Loss", color="tab:red")
ax1.plot(range(len(loss_history)), loss_history, color="tab:red", label="Training Loss")
ax1.tick_params(axis="y", labelcolor="tab:red")

# Plot validation loss (assuming it occurs every 'epoch_interval' steps)
epoch_interval = len(loss_history) // len(val_loss_history)
val_x = [
    i * epoch_interval for i in range(len(val_loss_history))
]  # X values for validation loss
ax1.plot(
    val_x,
    val_loss_history,
    color="tab:orange",
    marker="o",
    linestyle="dashed",
    label="Validation Loss",
)

# Create secondary y-axis for Learning Rate
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


# %% [markdown]
# # Model inference

# %%
def inference(input, tokenizer, model, max_length=TGT_MAX_SEQ):
    """
    Translates a single Polish sentence into Ukrainian using greedy decoding.
    """
    model.eval()  # Set the model to evaluation mode

    tokens = tokenizer.encode(input)
    tokens = torch.tensor(tokens).unsqueeze(0).to(device)  # Shape: (1, seq_len)
    # print(tokens)

    # Start with the input sentence and an empty target sequence
    src_data = tokens
    tgt_data = torch.tensor([tokenizer.bos_token_id]).unsqueeze(0).to(device)

    with torch.no_grad():
        for _ in range(max_length):
            output = model(src_data, tgt_data)

            next_token_id = output[:, -1, :].argmax(dim=-1).item()

            tgt_data = torch.cat([tgt_data, torch.tensor([[next_token_id]]).to(device)], dim=1)

            if next_token_id == tokenizer.eos_token_id:
                break

    # Decode the token ids back to the sentence
    translated_tokens = tgt_data.squeeze().tolist()
    translated_sentence = tokenizer.decode(translated_tokens, skip_special_tokens=True)

    return translated_sentence


def inference_from_datasets(train_dataset: bool = True, index: int = 0) -> (str, str):
    if train_dataset:
        dataset = train_loader.dataset
    else:
        dataset = val_loader.dataset
    src_input = tokenizer.decode(dataset[index][0].tolist(), skip_special_tokens=True)
    translation = inference(src_input, tokenizer, transformer)

    print('Dataset:', 'Train' if train_dataset else 'Validation')
    print('Src:', src_input)
    print('Generated translation:', translation)
    real_translation = tokenizer.decode(dataset[index][1].tolist(), skip_special_tokens=True)
    print('Real translation:      ', real_translation)
    return src_input, translation, real_translation


# %% [markdown]
# <!-- # SacreBLEU metrics -->

# %%
import evaluate
import httpx
# Load metrics
rouge = evaluate.load("rouge")
sacrebleu = evaluate.load("sacrebleu")
meteor = evaluate.load("meteor")
chrf = evaluate.load("chrf")
bleu = evaluate.load("bleu")

# %%
from dotenv import load_dotenv
import os

load_dotenv()  # Load variables from .env

deepl_auth_key = os.getenv("DEEPL_AUTH_KEY")

# %%

# Docs: https://huggingface.co/spaces/evaluate-metric/sacrebleu
# score from 0 to 100
local_accuracy = 0.0
deepl_accuracy = 0.0

def compute(predictions, references):
    # Predictions and references
    predictions = [predictions]
    references = [[references]]

    # Compute all metrics
    results_rouge = rouge.compute(predictions=predictions, references=references)  # ROUGE expects list of strings
    results_sacrebleu = sacrebleu.compute(predictions=predictions, references=references)
    results_meteor = meteor.compute(predictions=predictions, references=references)
    results_chrf = chrf.compute(predictions=predictions, references=references)
    results_bleu = bleu.compute(predictions=predictions, references=references)
    return results_rouge['rouge1']

    # Print results
    # print("\n📊 Translation Metrics:")
    # print(f"ROUGE-1 (0–1):        {results_rouge['rouge1']:.4f}")  # Overlap of unigrams (single words) between the generated and reference texts.
    # print(f"ROUGE-2 (0–1):        {results_rouge['rouge2']:.4f}")  # Overlap of bigrams (word pairs).
    # print(f"ROUGE-L (0–1):        {results_rouge['rougeL']:.4f}")  # Measures longest common subsequence (sequence similarity).
    # print(f"SacreBLEU (0–100):    {results_sacrebleu['score']:.2f}")  # Precision-based score for how many matching words/phrases, adjusted for brevity.
    # print(f"METEOR (0–1):         {results_meteor['meteor']:.4f}")  #  Considers word matches, stemming, and synonyms with penalties for word order.
    # print(f"chrF (0–100):         {results_chrf['score']:.2f}")  # Measures character-level n-gram overlap (more sensitive to small variations).
    # print(f"BLEU (0–100):         {results_bleu['bleu']:.2f}")  # Measures n-gram overlap between generated and reference texts.

n = 100
for i in range(n):
    src_input, predictions, references = inference_from_datasets(train_dataset=False, index=i)  # translation, real_translation


    json_data = {
        "text": [src_input], 
        "target_lang": "FR"
    }
    headers = {"Authorization": f"DeepL-Auth-Key {deepl_auth_key}"}
    response = httpx.post("https://api-free.deepl.com/v2/translate", json=json_data, headers=headers)
    response = response.json()["translations"][0]["text"]
    print('Translation from DeepL:', response)

    local_accuracy += compute(predictions, references)
    print()
    deepl_accuracy += compute(predictions, response)

print('Local accuracy:', local_accuracy/n)
print('DeepL accuracy:', deepl_accuracy/n)

# %%
# PATH = r"new_translation_model.pt"
# torch.save(transformer.state_dict(), PATH)

# %%
# next_model = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_src_seq_len,
#                          max_tgt_seq_len, dropout)
# next_model.load_state_dict(torch.load(PATH, weights_only=True))
# next_model = next_model.to(device)
# # print(next_model)

# # sentence = tokenizer.decode(train_loader.dataset[0][0].tolist(), skip_special_tokens=True)
# sentence = "What are light beans there?"
# print(sentence)
# # sentence = "Prehistoric humans studied the relationship between the seasons and the length of days to plan their hunting and gathering activities."
# translation = translate_sentence(sentence, tokenizer, next_model)
# print(translation)
