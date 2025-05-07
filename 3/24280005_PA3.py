# %% [markdown]
# # PA3 - LoRA implementation on SmolLM
# 
# ### Introduction
# 
# In this notebook, you will learn how to integrate LoRA (Low-Rank Adapters) into the SmolLM model you implemented in PA2. Before starting working on this notebook, please make sure to go through the README.md provided as it will intoduce to the concepts relevant to the assignment.
# 
# 
# ### Instructions
# 
# - Follow along with the notebook, filling out the necessary code where instructed.
# 
# - <span style="color: red;">Read the Submission Instructions, Plagiarism Policy, and Late Days Policy in the attached PDF.</span>
# 
# - <span style="color: red;">Make sure to run all cells for credit.</span>
# 
# - <span style="color: red;">Do not remove any pre-written code.</span>
# 
# - <span style="color: red;">You must attempt all parts.</span>

# %% [markdown]
# ### Imports

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import copy
import time
import os
from tqdm.notebook import tqdm
from tabulate import tabulate

# For tokenization and dataset loading
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# %% [markdown]
# #### Initializing device here for future use if needed

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% [markdown]
# #### The implementation for the SmolLM model from PA2 has been added below for your convenience. You just need to run the next 5 cells in order to define our model.

# %%
from dataclasses import dataclass

@dataclass
class smolConfig:
    vocab_size = 49152
    hidden_size = 576
    intermediate_size = 1536
    num_hidden_layers = 30
    num_heads = 9
    kv_heads = 3

# %%
torch.manual_seed(42)

def rotate_half(x):
    """
    Helper function to rotate the left half of a tensor along its final dimension.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]

    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """
    Applies RoPE on the query and key tensors.
    """
    cos, sin = cos.to(q.device), sin.to(q.device)

    # Unsqueexzing to enable broadcasting
    sin = sin.unsqueeze(unsqueeze_dim)
    cos = cos.unsqueeze(unsqueeze_dim)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed

class RotaryEmbedder(nn.Module):
    def __init__(self, dim, base):
        super().__init__()
        # Precompute frequency for sine/cosine embeddings
        self.freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

    @torch.no_grad()
    def forward(self, x):
        # Generate positions (sequence indices) for the input
        pos = torch.arange(x.shape[-2], dtype=torch.long)
        # Compute angles for sine and cosine embeddings
        angles = torch.einsum("p,f->pf", pos.float(), self.freq).unsqueeze(dim=0)
        # Duplicate angles for sine and cosine embeddings
        emb = torch.cat((angles, angles), dim=-1)
        # Return cosine and sine components of the positional embeddings
        return emb.cos(), emb.sin()

class GroupedQueryAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Model dimensions and attention configurations
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.kv_heads = config.kv_heads  # Number of key-value heads
        self.rope_theta = 10000.0  # Scaling factor for rotary embeddings

        # Linear projections for queries, keys, values, and output
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        # Rotary embedding generator
        self.rotary_emb = RotaryEmbedder(base=self.rope_theta, dim=self.head_dim)

    def _repeat_kv(self, x, n_rep):
        batch, num_key_value_heads, slen, head_dim = x.shape
        # Expand the number of key-value heads by repeating them
        x = x[:, :, None, :, :].expand(
            batch, num_key_value_heads, n_rep, slen, head_dim
        )
        # Reshape to align with the expected multi-head attention format
        return x.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

    def forward(self, x: torch.Tensor, attention_mask=None):
        # Input dimensions: (batch_size, seq_len, hidden_size)
        b, q, _ = x.size()

        # Project input hidden states into queries, keys, and values
        q_states = self.q_proj(x)
        k_states = self.k_proj(x)
        v_states = self.v_proj(x)

        # Reshape and transpose for multi-head attention
        q_states = q_states.view(b, q, self.num_heads, self.head_dim).transpose(1, 2)
        k_states = k_states.view(b, q, self.kv_heads, self.head_dim).transpose(1, 2)
        v_states = v_states.view(b, q, self.kv_heads, self.head_dim).transpose(1, 2)

        # Compute rotary positional embeddings
        cos, sin = self.rotary_emb(q_states)
        cos = cos.to(q_states.device)
        sin = sin.to(q_states.device)
        # Apply positional embeddings to queries and keys
        q_states, k_states = apply_rotary_pos_emb(q_states, k_states, cos, sin)

        # Repeat key and value tensors to match the number of query heads
        __kv_groups = self.num_heads // self.kv_heads
        k_states = self._repeat_kv(k_states, __kv_groups)
        v_states = self._repeat_kv(v_states, __kv_groups)

        # Compute attention scores (scaled dot-product attention)
        attn_weights = torch.matmul(q_states, k_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        # Add attention mask (e.g., for causal or padding masking)
        attn_weights = attn_weights + attention_mask

        # Normalize attention weights using softmax
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        # Apply dropout to attention weights
        attn_weights = nn.functional.dropout(attn_weights, 0)

        # Compute attention output
        attn_output = torch.matmul(attn_weights, v_states)
        # Reshape and transpose back to original format
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(b, q, -1)

        # Project the attention output back to the hidden size
        attn_output = self.o_proj(attn_output)

        # Return the final attention output
        return attn_output

# %%
torch.manual_seed(42)

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        This is the Root Mean Square Normalisation class.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))  # Learnable scaling factor
        self.variance_epsilon = eps

    def forward(self, x):
        # Calculate variance along the last dimension (hidden size)
        variance = x.pow(2).mean(-1, keepdim=True)

        # Normalize and scale
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * x

class MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        """
        This is the gated MLP from the LLaMa architecture. Here we use the SiLU acitvation.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.activation = nn.modules.activation.SiLU()

        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)

    def forward(self, x):
        down_proj = self.down_proj(self.activation(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

class LlamaDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        """
        This is the Llama decoder block.
        """
        # Self Attention Module
        self.self_attn = GroupedQueryAttention(config)

        # FFN Module
        self.mlp = MLP(hidden_size=config.hidden_size, intermediate_size=config.intermediate_size)

        # Pre Attention and Post Attention normalisation
        self.input_layernorm = RMSNorm(config.hidden_size, eps=1e-05)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=1e-05)

    def forward(self, x, attention_mask):
        # Skip connection cache

        residual = x

        # Pre-attention normalisation
        x = self.input_layernorm(x)

        # A causal attention mask (i.e., decoder can only look at tokens that it has generated thus far)
        attention_mask = torch.triu(torch.full((attention_mask.shape[-1], attention_mask.shape[-1]),
                                               fill_value=float('-inf')), diagonal=1)

        attention_mask = attention_mask.to(x.device)

        # Self-attention block
        x = self.self_attn(x=x,attention_mask=attention_mask)
        x += residual

        # Skip connection cache for MLP
        residual = x

        # Pre-MLP normalisation
        x = self.post_attention_layernorm(x)

        # MLP block
        x = self.mlp(x)
        x += residual

        return x

# %%
torch.manual_seed(42)

class smolModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # embedding layer which maps each token to a vector embedding
        self.embed_tokens = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size
        )

        # Stack of decoder layers (LlamaDecoder) defined by the configuration
        self.layers = nn.ModuleList([
            LlamaDecoder(config) for _ in range(config.num_hidden_layers)
        ])

        # RMSNorm: final layer normalization applied to hidden states
        self.norm = RMSNorm(config.hidden_size, eps=1e-05)

    def forward(self, input_ids=None, attention_mask=None):
        inputs_embeds = self.embed_tokens(input_ids)
        x = inputs_embeds

        # Pass embeddings through each decoder layer
        for i, decoder_layer in enumerate(self.layers):
            layer_outputs = decoder_layer(
                x,
                attention_mask=attention_mask
            )
            x = layer_outputs

        # Final normalisation
        x = self.norm(x)

        return x

class smolLM(nn.Module):
    """
    This is the Language Model.
    It passes the embeddings from the SmolLM backbone into a LM head.
    The LM head generates logits over the space of the entire vocabulary for next word prediction.
    """
    def __init__(self, config):
        super().__init__()
        # SmolLM backbone which generates the contextualised embeddings for the input tokens
        self.model = smolModel(config)
        # The LM head which maps embeddings to logits over the vocabulary
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # weights between LM head and the token_embedding layer are shared in the SmolLM architecture
        self.tie_weights()

    def tie_weights(self):
        # lm_head shares weights with the embedding layer
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids, attention_mask):
        # Input tokens are passed to the SmolLM backbone
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # embeddings corresponding to each input token => (batch_size, seq_len, emb_dim)
        x = outputs

        # pass the embeddings through the LM head
        logits = self.lm_head(x).float()
        return {'logits': logits}

# %%
def __generate(model, inputs, num_tokens, tokenizer, max_length=50):
    """
    A basic greedy approach for text generation.
    """
    collect = []
    for _ in range(num_tokens):
        output = model(**inputs)
        output_id = torch.argmax(output['logits'][0, -1]).item()
        collect.append(output_id)
        if output_id == tokenizer.eos_token_id or len(collect) >= max_length:
            break
        # Update input_ids and attention_mask
        new_token = torch.tensor([output_id], device=inputs['input_ids'].device)
        inputs['input_ids'] = torch.cat([inputs['input_ids'][0], new_token]).unsqueeze(0)
        inputs['attention_mask'] = F.pad(inputs['attention_mask'], (0, 1), value=1)
    return tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(collect))

# %% [markdown]
# ### Understanding the Problem
# 
# In this assignment, we'll implement LoRA, which allows efficient fine-tuning by adding low-rank decomposition matrices to specific weight matrices in the model. For each targeted weight matrix $W \in \mathbb{R}^{d \times k}$, we'll create two smaller matrices $A \in \mathbb{R}^{r \times k}$ and $B \in \mathbb{R}^{d \times r}$ where $r \ll \min(d, k)$.
# 
# The key equation is:
# $W' = W + \frac{\alpha}{r}BA$

# %% [markdown]
# ### Base LoRA Implementation
# First, we'll implement a generic LoRA module that can be applied to any linear layer in our model.

# %%
torch.manual_seed(42)

class LoRALayer(nn.Module):
    """
    Implementation of a LoRA layer - a low-rank adaptation of a weight matrix.
    """

    def __init__(self, in_features, out_features, rank=8, alpha=16, dropout=0.0):
        """
        Initialize a LoRA layer.
        """
        super().__init__()
        # scaling factor
        self.scaling = alpha/rank

        # matrix A: According to the paper, A is initialized with gausian distribution
        self.A = nn.Parameter(torch.randn((rank,out_features)))

        # matrix B: According to the paper, B is initialized with zeros
        self.B = nn.Parameter(torch.zeros((in_features,rank)))

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x:torch.Tensor):
        ## implement the forward pass here
        #print(x.shape)  # torch.Size([1, 5, 576])
        output = self.scaling * (self.dropout(x) @ torch.matmul(self.B,self.A))

        return output

# %% [markdown]
# **Question 1**: Why is it important to initialize matrix B with zeros? How does this affect training at the beginning?
# 
# <div style="color:green">
# 
# **Ans**: $B$ is initialized with zeros to make the product $\nabla W = BA = \mathbf{0}$. This is because the product represents low rank weights adaptation and since at the beginning of the training there have been no weight updates, this matrix is initialized with zeroes. This allows the model to stabilize at the beginning of training, adapt over time, and avoid drastic changes that could arise from both matrices being initialized with random values.
# </div>

# %% [markdown]
# ### LoRA-Enhanced Linear Layer
# 
# Now we'll create a wrapper for linear layers that incorporates LoRA:

# %%
torch.manual_seed(42)

class LoRALinear(nn.Module):
    """
    A linear layer with LoRA adaptation.
    """

    def __init__(self, linear_layer:nn.Linear, rank=8, alpha=16, dropout=0.0):
        """
        Initialize a LoRA-adapted linear layer.
        """
        super().__init__()
        # original linear layer
        self.linear = linear_layer

        # freeze the weights of the original layer
        for param in self.linear.parameters():
            param.requires_grad = False

        # input and output dimensions from the linear layer
        in_features = self.linear.in_features
        out_features = self.linear.out_features

        # create the LoRA adaptation layer
        self.lora = LoRALayer(in_features, out_features, rank, alpha, dropout)

    def forward(self, x):
        ## implement the forward pass here

        output = self.linear(x) + self.lora(x)
        return output

# %% [markdown]
# ### Applying LoRA to SmolLM
# 
# **LoRA Integration Strategy**\
# We need to decide which weights in our model should be adapted with LoRA. In transformers, typical targets include:
# - Query, Key, Value projections in attention layers
# - Output projections in attention layers
# - Up/down projections in feed-forward networks
# 
# Implement the function to add LoRA to specific linear layers in the model.

# %%
def add_lora_to_model(model: smolLM, target_modules=None, rank=8, alpha=16, dropout=0.0):
    """
    Add LoRA adapters to target modules in the model.

    Returns:
        Model with LoRA adapters
    """
    ## your code here:
    if target_modules is None:
         return model
    model_with_lora = model

    for param in model_with_lora.parameters():
         param.requires_grad = False

    for name, module in model_with_lora.named_modules():
            #print(name)
            if isinstance(module, nn.Linear):
                if any(target in name for target in target_modules):
                    adaptation = LoRALinear(module,rank, alpha, dropout)
                    for params in adaptation.lora.parameters():
                        param.requires_grad = True
                    model_with_lora.set_submodule(name,adaptation)

                    #print(f'Replaced {name}')
    return model_with_lora

# %% [markdown]
# **Initializing the Base and LoRA models.**

# %%
config = smolConfig()
base_model = smolLM(config)
checkpoint = "HuggingFaceTB/SmolLM-135M"
reference_model = AutoModelForCausalLM.from_pretrained(checkpoint)
base_model.load_state_dict(reference_model.state_dict(), strict=False)

target_modules = [
    'q_proj',
    'k_proj',
    'v_proj',
    'o_proj',
    'up_proj',
    'down_proj'
]

## DO NOT CHANGE THIS
lora_model = add_lora_to_model(
    base_model,
    target_modules=target_modules,
    rank=4,
    alpha=8,
    dropout=0.3,
)

# %%
lora_model

# %% [markdown]
# ### Parameter Analysis
# 
# Let's compare the parameter counts between the original model and the LoRA-enhanced version. Implement the parameter counting and analysis function.
# 
# You should see that the % of trainable parameters in our `lora_model` should be <1%.

# %%
def count_parameters(model: nn.Module, only_trainable: bool = False):
    """
    Count the number of parameters in a model.

    Args:
        model: PyTorch model
        only_trainable: If True, count only trainable parameters

    Returns:
        Number of parameters
    """
    if only_trainable:
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def analyze_parameters(original_model: nn.Module, lora_model: nn.Module):
    """
    Analyze parameter counts between original and LoRA-adapted models.

    Returns:
        Dictionary with parameter statistics
    """

    total_params = count_parameters(original_model)
    trainable_params = count_parameters(lora_model, only_trainable=True)

    # calculate parameter savings
    param_percent = (trainable_params / total_params) * 100

    # count parameters by layer type
    lora_params_by_type = {}
    for name, module in lora_model.named_modules():
        if isinstance(module, LoRALayer):
            # extract the module type from the name
            parts = name.split(".")
            module_type = next(
                (
                    p
                    for p in parts
                    if any(
                        t in p
                        for t in [
                            "q_proj",
                            "k_proj",
                            "v_proj",
                            "o_proj",
                            "up_proj",
                            "down_proj"
                        ]
                    )
                ),
                "other",
            )

            # count parameters in this LoRA layer
            params = sum(p.numel() for p in module.parameters())

            # add to the count by type
            if module_type in lora_params_by_type:
                lora_params_by_type[module_type] += params
            else:
                lora_params_by_type[module_type] = params

    stats =  {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "param_percent": param_percent,
        "params_by_type": lora_params_by_type,
    }

    return stats

stats = analyze_parameters(base_model, lora_model)

print(f"Total Parameters in Original Model: {stats['total_params']}")
print(f"Trainable Parameters in LoRA Model: {stats['trainable_params']}")
print(f"% of trainable parameters: {stats['param_percent']:.2f}%")
print()
print(f"LoRA Parameters in each layer:")
for k, v in stats['params_by_type'].items():
    print(f"{k}: {v}")

# %% [markdown]
# ### Fine-tuning with LoRA

# %% [markdown]
# #### Dataset Preparation
# Let's set up a small dataset for fine-tuning:

# %%
def prepare_dataset(
    tokenizer,
    dataset_name="databricks/databricks-dolly-15k",
    subset=None,
    max_samples=500,
):
    """
    Prepare a dataset for fine-tuning.

    Args:
        tokenizer: Tokenizer to use
        dataset_name: HuggingFace dataset name
        subset: Dataset subset (if applicable)
        max_samples: Maximum number of samples to use

    Returns:
        Processed dataset ready for training
    """
    # load dataset
    if subset:
        dataset = load_dataset(dataset_name, subset)
    else:
        dataset = load_dataset(dataset_name)

    train_data = (
        dataset["train"]
        .shuffle(seed=42)
        .select(range(min(max_samples, len(dataset["train"]))))
    )

    train_val_split = train_data.train_test_split(test_size=0.2, seed=42)
    train_data = train_val_split["train"]
    val_data = train_val_split["test"]

    def tokenize_function(examples):
        tokenized = tokenizer(examples["instruction"], padding="max_length", truncation=True, max_length=512)
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    train_tokenized = train_data.map(
        tokenize_function, batched=True, remove_columns=train_data.column_names
    )
    val_tokenized = val_data.map(
        tokenize_function, batched=True, remove_columns=val_data.column_names
    )

    train_tokenized.set_format("torch")
    val_tokenized.set_format("torch")

    train_dataloader = DataLoader(train_tokenized, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_tokenized, batch_size=8)

    return train_dataloader, val_dataloader

# %% [markdown]
# #### Initialzing our Tokenizer and Dataset.

# %%
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# the tokenizer does not have a defined padding token, so we initialize our own as the [EOS] token.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

train_dataloader, val_dataloader = prepare_dataset(tokenizer=tokenizer,  dataset_name="databricks/databricks-dolly-15k", max_samples=3000)

# %% [markdown]
# **We can test our base model to ensure it's working correctly.**

# %%
prompt = "The future of AI is"
inputs = tokenizer(prompt, return_tensors="pt")

out = __generate(base_model, inputs, num_tokens=100, tokenizer=tokenizer)

print('=='*10 + f' Output generated' + '=='*10)
print(prompt + ' ' + out)

# %% [markdown]
# #### Training Loop
# The training function for our model with LoRA adapters has been implemented below

# %%
def train_lora(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    epochs=3,
    device=None,
):
    """
    Train a model with LoRA adapters.

    Args:
        model: LoRA-adapted model
        train_dataloader: Training data
        val_dataloader: Validation data
        optimizer: PyTorch optimizer
        epochs: Number of training epochs
        device: Device to train on

    Returns:
        Training history
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_perplexity": [],
        "val_perplexity": [],
    }

    # add scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    for epoch in range(epochs):
        model.train()
        train_losses = []

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} [Train]")

        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            shift_attention_mask = attention_mask[:, 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss(reduction="none")
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

            loss = loss.view(shift_labels.size())
            loss = (loss * shift_attention_mask).sum() / shift_attention_mask.sum()

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            progress_bar.set_postfix({"train_loss": loss.item()})

        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_train_perplexity = torch.exp(torch.tensor(avg_train_loss)).item()

        model.eval()
        val_losses = []

        progress_bar = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{epochs} [Val]")

        with torch.no_grad():
            for batch in progress_bar:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs["logits"]

                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                shift_attention_mask = attention_mask[:, 1:].contiguous()

                loss_fct = nn.CrossEntropyLoss(reduction="none")
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
                )

                loss = loss.view(shift_labels.size())
                loss = (loss * shift_attention_mask).sum() / shift_attention_mask.sum()

                val_losses.append(loss.item())
                progress_bar.set_postfix({"val_loss": loss.item()})

        avg_val_loss = sum(val_losses) / len(val_losses)
        avg_val_perplexity = torch.exp(torch.tensor(avg_val_loss)).item()

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_perplexity"].append(avg_train_perplexity)
        history["val_perplexity"].append(avg_val_perplexity)

        print(
            f"Epoch {epoch+1}/{epochs} - "
            f"Train Loss: {avg_train_loss:.4f}, Train Perplexity: {avg_train_perplexity:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, Val Perplexity: {avg_val_perplexity:.4f}"
        )

        scheduler.step()

    return history, model

# %% [markdown]
# #### Training our Model.
# **DO NOT MODIFY THE HYPERPARAMETERS**

# %%
## DO NOT CHANGE THIS

optimizer = torch.optim.AdamW(
    [p for p in lora_model.parameters() if p.requires_grad], lr=1e-4, weight_decay=0.01
)

history, trained_lora_model = train_lora(model=lora_model, train_dataloader=train_dataloader, val_dataloader=val_dataloader, optimizer=optimizer, epochs=5)

# %% [markdown]
# **Optional: You can save your trained model in case you decide to do the assignment in parts.**

# %%
torch.save(trained_lora_model.state_dict(), "lora_finetuned_model.pth")

# %% [markdown]
# #### Merging LoRA Weights for Inference
# For efficient inference, we can merge LoRA weights with the original weights.
# 
# Implement the function `merge_lora_weights`.

# %%
def merge_lora_weights(model):
  """
  Merge LoRA weights with original weights for efficient inference.

  Args:
      model: LoRA-adapted model

  Returns:
      Model with merged weights
  """
  merged_model = copy.deepcopy(model)

  for name, module in merged_model.named_modules():
    if isinstance(module, LoRALinear):
      # Grab the frozen linear layer and the LoRA weights
      linear = module.linear
      A = module.lora.A  # (rank, out_features)
      B = module.lora.B  # (in_features, rank)
      scaling = module.lora.scaling

      #  W + scaling * (B @ A) - merged weighgt
      merged_weight = linear.weight.data + scaling * (B @ A).T   # data is a tensor, weight is parameter
      merged_linear = nn.Linear(linear.in_features, linear.out_features, bias=linear.bias is not None)
      merged_linear.weight.data = merged_weight

      if linear.bias is not None:
          merged_linear.bias.data = linear.bias.data.clone()

      # Replace the LoRALinear with the merged Linear
      merged_model.set_submodule(name, merged_linear)
  return merged_model

# %%
# Merge LoRA weights into the base model
merged_model = merge_lora_weights(trained_lora_model)

# %% [markdown]
# **Optional: Save your merged model.**

# %%
torch.save(merged_model.state_dict(), "merged_lora_model.pth")

# %% [markdown]
# ### Text Generation and Comparison
# Now let's compare text generation between models.

# %% [markdown]
# #### Loading in the fully finetuned model.
# Instead of having you fully finetune the model, we are sharing the weights to make your life a little easier. First, we'll load in our fully finetuned model. The weights are accesible through the drive link [Finetuned Base Model Weights](https://drive.google.com/drive/folders/1eIflNAp9UE4Fm8ZrBAzjDPsOCs-s_O55?usp=sharing)
# 

# %%
from google.colab import drive
drive.mount('/content/drive')

# %%
finetuned_base_model = smolLM(config)

## add your model path here
model_path = "/content/drive/MyDrive/GenAI/full_finetuned_smolLM.pth"

# Load the finetuned weights into the base model
finetuned_base_model.load_state_dict(torch.load(model_path, weights_only=True))

# Set to eval mode for inference
finetuned_base_model.eval()

# %% [markdown]
# #### We can now compare the fully finetuned and LoRA finetuned model to evaluate the effectiveness of using LoRA.

# %%
def compare_generations(models, tokenizer, prompts, max_tokens=100):
    """
    Compare text generation between different model versions.

    Args:
        models: Dictionary of models to compare
        tokenizer: Tokenizer
        prompts: List of prompts to test
        max_tokens: Maximum tokens to generate

    Returns:
        DataFrame with generation results
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []

    def calculate_perplexity(model, inputs):
        """
        Computes perplexity for a given model and input.
        """
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs["logits"]
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = inputs["input_ids"][:, 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(reduction="mean")
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            perplexity = torch.exp(loss).item()
        return perplexity

    for prompt in prompts:
        print(f"Prompt: {prompt}")
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        prompt_results = {"Prompt": prompt}

        for model_name, model in models.items():
            model.to(device)
            model.eval()

            start_time = time.time()
            output = __generate(
                model, inputs.copy(), num_tokens=max_tokens, tokenizer=tokenizer
            )
            end_time = time.time()

            perplexity = calculate_perplexity(model, inputs)

            prompt_results[f"{model_name} Perplexity"] = perplexity

            print(f"Model: {model_name}")
            print(f"Generated: {output}")
            print(f"Time: {end_time - start_time:.2f}s")
            print(f"Perplexity: {perplexity:.4f}")
            print("-" * 50)

        results.append(prompt_results)

    df_results = pd.DataFrame(results)

    return df_results

# %%
# Define models for comparison
models = {
    "Fully Finetuned Model": finetuned_base_model,
    "LoRA Finetuned Model": merged_model,
}

# Define prompts to test
prompts = [
    "Once upon a time, in a distant galaxy,",
    "The future of artificial intelligence is",
    "A wise old wizard once said,",
]

# Run the comparison
df_results = compare_generations(models, tokenizer, prompts, max_tokens=100)

# %% [markdown]
# **Compare the perplexity scores of the models**
# 

# %%
print(tabulate(df_results, headers='keys', tablefmt='fancy_grid'))

# %% [markdown]
# ### Analysis and Discussion
# For this section, analyze your results and answer the following questions:
# 
# **Question 2:** How does LoRA performance compare to full fine-tuning? What are the tradeoffs?
# 
# 
# <div style="color:green">
# 
# **Ans:**
# 
# LoRA model gives lower perplexity scores than fully finetuned model, meaning the lora module is better at predicting the correct next tokens.
# 
# LoRA is gives better performance with fewer trainable parameters, less compute, quicker training time resulting in easy scalability and faster and cheaper deployment.
# 
# However, the tradeoff is that LoRA only modifies a small subset of parameters (typically within attention or feedforward layers), which might limit its expressiveness for certain tasks. Full fine-tuning, while more expensive, allows the model to adjust all weights, which can be advantageous when major shifts in behavior or representation are required,especially in domains very different from the pretraining data.
# 
# </div>
# 
# 
# **Question 3:** Which target modules benefit most from LoRA adaptation in SmolLM?
# 
# <div style="color:green">
# 
# **Ans:** The attention modules because there were 30 decoders stacked and lora significantly reduced the rank of the query, key and value projections over 30 decoder stacks.
# 
# </div>
# 
# **Question 4:** How does rank value affect the quality of adaptation and the parameter count?
# 
# <div style="color:green">
# 
# **Ans:**
# Higher rank provides more complex adaptation and improve perplexity score, but involve more trainable parametrs. The results however will be better because the high rank will give it more capacity to capture gradient changes.
# 
# </div>
# 
# **Question 5:** What are the practical benefits of LoRA for deploying fine-tuned models?
# 
# <div style="color:green">
# 
# **Ans:**
# Lora provides models having fewer trainanable parameters. It allows efficient fine-tuning with minimal compute and storage requirements. So, it provides fast deployment, and less resource overhead when switching adapters.
# 
# </div>

# %% [markdown]
# # Fin.


