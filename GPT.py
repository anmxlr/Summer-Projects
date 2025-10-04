
import numpy as np
import PyPDF2
import sys
from math import sqrt

def extract_text_from_pdf(pdf_path: str) -> str:
    text_accumulator = []
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_accumulator.append(page_text)
    return "\n".join(text_accumulator)

def read_text_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


class CharacterTokenizer:
    def __init__(self, text: str):
        unique_chars = sorted(list(set(text)))
        self.char_to_index = {ch: i for i, ch in enumerate(unique_chars)}
        self.index_to_char = {i: ch for ch, i in self.char_to_index.items()}
        self.vocab_size = len(unique_chars)

    def encode(self, text: str) -> np.ndarray:
        return np.array([self.char_to_index[ch] for ch in text], dtype=np.int32)

    def decode(self, indices: np.ndarray) -> str:
        return ''.join([self.index_to_char[int(i)] for i in indices])



def stable_softmax(matrix: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = matrix - np.max(matrix, axis=axis, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=axis, keepdims=True)

def one_hot(indices: np.ndarray, depth: int) -> np.ndarray:
    flat = indices.reshape(-1)
    mat = np.zeros((flat.shape[0], depth), dtype=np.float64)
    mat[np.arange(flat.shape[0]), flat] = 1.0
    return mat.reshape(indices.shape + (depth,))


def layer_norm_forward(x: np.ndarray, epsilon: float = 1e-5):
    # x: (batch, seq_len, dim)
    mean = np.mean(x, axis=-1, keepdims=True)           # μ
    variance = np.var(x, axis=-1, keepdims=True)        # σ^2
    std_inv = 1.0 / np.sqrt(variance + epsilon)         # 1 / sqrt(σ^2 + ε)
    normalized = (x - mean) * std_inv                   # y
    cache = (normalized, std_inv)
    return normalized, cache

def layer_norm_backward(grad_normalized: np.ndarray, cache):
    # grad_normalized: dL/dy same shape as x
    normalized, std_inv = cache
    # Let N = dim
    N = normalized.shape[-1]
    # Using vectorized formula for dL/dx:
    # dL/dx = (1/std) * (dL/dy - mean(dL/dy) - normalized * mean(dL/dy * normalized))
    grad_mean = np.mean(grad_normalized, axis=-1, keepdims=True)
    grad_norm_dot_norm = np.mean(grad_normalized * normalized, axis=-1, keepdims=True)
    grad_x = std_inv * (grad_normalized - grad_mean - normalized * grad_norm_dot_norm)
    return grad_x


class MultiHeadCausalSelfAttention:
    def __init__(self, model_dim: int, num_heads: int, random_seed: int = 0):
        assert model_dim % num_heads == 0, "model_dim must be divisible by num_heads"
        np.random.seed(random_seed)
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads

        # Initialize parameter matrices
        scale = 1.0 / sqrt(model_dim)
        self.W_query = np.random.randn(model_dim, model_dim) * scale    # (D, D)
        self.W_key   = np.random.randn(model_dim, model_dim) * scale
        self.W_value = np.random.randn(model_dim, model_dim) * scale
        self.W_output= np.random.randn(model_dim, model_dim) * scale

        # Placeholders for gradients to be filled in backward pass
        self.gradient_W_query = np.zeros_like(self.W_query)
        self.gradient_W_key = np.zeros_like(self.W_key)
        self.gradient_W_value = np.zeros_like(self.W_value)
        self.gradient_W_output = np.zeros_like(self.W_output)

    def forward(self, input_tensor: np.ndarray):
        """
        input_tensor: (B, T, D)
        returns: output_tensor (B, T, D) and stores caches for backward
        """
        batch_size, sequence_length, model_dim = input_tensor.shape
        # Linear projections
        Q = input_tensor @ self.W_query    # (B, T, D)
        K = input_tensor @ self.W_key
        V = input_tensor @ self.W_value

        # Reshape to (B, num_heads, T, head_dim)
        Q_heads = Q.reshape(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(0,2,1,3)
        K_heads = K.reshape(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(0,2,1,3)
        V_heads = V.reshape(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(0,2,1,3)

        # Scaled dot-product scores: (B, num_heads, T, T)
        scale_factor = 1.0 / sqrt(self.head_dim)
        raw_scores = (Q_heads @ K_heads.transpose(0,1,3,2)) * scale_factor

        # Apply causal mask: prevent attention to future positions
        causal_mask = np.triu(np.ones((sequence_length, sequence_length), dtype=bool), k=1)
        raw_scores_masked = np.where(causal_mask[None, None, :, :], -1e9, raw_scores)

        # Softmax over last axis (keys/time)
        attention_weights = stable_softmax(raw_scores_masked, axis=-1)   # (B, H, T, T)

        # Attention output
        attention_output_heads = attention_weights @ V_heads             # (B, H, T, head_dim)

        # Merge heads -> (B, T, D)
        attention_output = attention_output_heads.transpose(0,2,1,3).reshape(batch_size, sequence_length, model_dim)

        # Final linear projection
        output_tensor = attention_output @ self.W_output    # (B, T, D)

        # Save cache for backward
        self.cache = {
            "input_tensor": input_tensor,
            "Q": Q, "K": K, "V": V,
            "Q_heads": Q_heads, "K_heads": K_heads, "V_heads": V_heads,
            "raw_scores": raw_scores, "raw_scores_masked": raw_scores_masked,
            "attention_weights": attention_weights,
            "attention_output_heads": attention_output_heads,
            "attention_output": attention_output
        }

        return output_tensor

    def backward(self, upstream_gradient: np.ndarray):
        """
        upstream_gradient: dL/d(output_tensor) shape (B, T, D)
        Computes gradients for:
            - input_tensor (dL/dX)
            - parameter matrices W_query, W_key, W_value, W_output
        Stores gradients in object's gradient_* fields and returns dL/dX.
        """
        cache = self.cache
        X = cache["input_tensor"]
        batch_size, sequence_length, model_dim = X.shape

        # grad w.r.t W_output from upstream: attention_output.T @ d_out
        attention_output = cache["attention_output"]   # (B, T, D)
        grad_W_output = attention_output.reshape(batch_size*sequence_length, model_dim).T @ upstream_gradient.reshape(batch_size*sequence_length, model_dim)
        # dAttentionOutput = dOut @ W_output^T
        grad_attention_output = upstream_gradient @ self.W_output.T   # (B,T,D)

        # reshape grads to heads: (B, H, T, head_dim)
        grad_attention_output_heads = grad_attention_output.reshape(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(0,2,1,3)
        # attention_output_heads = attention_weights @ V_heads
        attention_weights = cache["attention_weights"]  # (B,H,T,T)
        V_heads = cache["V_heads"]                      # (B,H,T,head_dim)

        # Grad w.r.t attention_weights: dA = dAttentionOutputHeads @ V_heads^T
        grad_attention_weights = grad_attention_output_heads @ V_heads.transpose(0,1,3,2)  # (B,H,T,T)
        # Grad w.r.t V_heads: dV = attention_weights^T @ dAttentionOutputHeads
        grad_V_heads = attention_weights.transpose(0,1,3,2) @ grad_attention_output_heads  # (B,H,T,head_dim)
        # but easier: grad_V_heads = attention_weights @ grad_attention_output_heads? careful:
        # Actually attention_output_heads = attention_weights @ V_heads
        # So dV_heads = attention_weights.transpose(...,1,0)??? Simpler: use matmul along last two axes:
        # We compute properly:
        grad_V_heads = attention_weights.transpose(0,1,2,3) @ grad_attention_output_heads  # (B,H,T,head_dim)
        # (Note: above line replicates attention_weights @ grad_attention_output_heads - shapes align)

        # Now we need gradient through softmax: raw_scores_masked -> attention_weights
        raw_scores = cache["raw_scores"]               # (B,H,T,T)
        raw_scores_masked = cache["raw_scores_masked"] # (B,H,T,T)
        A = attention_weights                          # (B,H,T,T)
        # dRawScores = dA * Jacobian_of_softmax
        # For each row (length T), Jacobian J where J_ij = A_i (δ_ij - A_j)
        # We compute efficiently: for each (b,h,t, :), dRawScores = (dA - sum(dA * A, axis=-1,keepdims=True)) * A
        temp = grad_attention_weights
        temp_sum = np.sum(temp * A, axis=-1, keepdims=True)    # (B,H,T,1)
        grad_raw_scores_masked = (temp - temp_sum) * A         # (B,H,T,T)

        # Masked positions had -1e9, their softmax contributions approx zero; grad there is zero as A ~ 0.
        # gradient flows unchanged (we already used A which is ~0 there).

        # scaled raw_scores = (Q @ K^T) * (1/sqrt(head_dim))
        scale_factor = 1.0 / sqrt(self.head_dim)
        grad_raw_scores = grad_raw_scores_masked * 1.0  # same shape

        # dQ_heads = grad_raw_scores @ K_heads
        K_heads = cache["K_heads"]
        grad_Q_heads = grad_raw_scores @ K_heads  # (B,H,T,head_dim)
        # dK_heads = grad_raw_scores.transpose(...,2,3?) careful: raw_scores = Q @ K^T, so dK = raw_scores^T @ dRawScores^T?
        grad_K_heads = grad_raw_scores.transpose(0,1,3,2) @ cache["Q_heads"]  # (B,H,T,head_dim)
        # both multiplied by scale factor
        grad_Q_heads *= scale_factor
        grad_K_heads *= scale_factor

        # Now we have grad_V_heads already from above (grad_V_heads)
        # Merge head grads back to (B,T,D)
        grad_Q = grad_Q_heads.transpose(0,2,1,3).reshape(batch_size, sequence_length, model_dim)
        grad_K = grad_K_heads.transpose(0,2,1,3).reshape(batch_size, sequence_length, model_dim)
        grad_V = grad_V_heads.transpose(0,2,1,3).reshape(batch_size, sequence_length, model_dim)

        # Compute gradients w.r.t parameter matrices W_query, W_key, W_value
        grad_W_query = (X.reshape(batch_size*sequence_length, model_dim).T @ grad_Q.reshape(batch_size*sequence_length, model_dim))
        grad_W_key   = (X.reshape(batch_size*sequence_length, model_dim).T @ grad_K.reshape(batch_size*sequence_length, model_dim))
        grad_W_value = (X.reshape(batch_size*sequence_length, model_dim).T @ grad_V.reshape(batch_size*sequence_length, model_dim))

        # Gradient w.r.t input X from these three projections
        grad_from_Q = grad_Q @ self.W_query.T
        grad_from_K = grad_K @ self.W_key.T
        grad_from_V = grad_V @ self.W_value.T

        grad_input_from_projections = grad_from_Q + grad_from_K + grad_from_V  # (B,T,D)

        # Also gradient flows from grad_attention_output through W_output back to attention_output
        grad_attention_output_flat = grad_attention_output.reshape(batch_size*sequence_length, model_dim)
        grad_W_output = (attention_output.reshape(batch_size*sequence_length, model_dim).T @ upstream_gradient.reshape(batch_size*sequence_length, model_dim))  # already computed above

        # Save computed gradients in the object
        self.gradient_W_query = grad_W_query
        self.gradient_W_key = grad_W_key
        self.gradient_W_value = grad_W_value
        self.gradient_W_output = grad_W_output

        # Total gradient to input: sum of projection gradients + gradient coming via W_output path already accounted as grad_from_attention_output? We already included grad_attention_output path by grad_attention_output variable.
        # grad_input = grad_input_from_projections + grad_from_attention_output? Wait: grad_attention_output was mapped to W_output gradient; grad_attention_output_heads was computed from grad_attention_output which eventually contributed to grad_V_heads etc. We already included that path via grad_attention_output_heads -> grad_V_heads/Q_heads/etc. So the only remaining grad is grad_input_from_projections.
        grad_input = grad_input_from_projections

        return grad_input

# ---------------------------
# Equations:
#   h1 = X W1 + b1
#   h2 = ReLU(h1)
#   output = h2 W2 + b2
# ---------------------------
class PositionwiseFeedForward:
    def __init__(self, model_dim: int, hidden_dim: int, random_seed: int = 0):
        np.random.seed(random_seed)
        scale1 = 1.0 / sqrt(model_dim)
        scale2 = 1.0 / sqrt(hidden_dim)
        self.W1 = np.random.randn(model_dim, hidden_dim) * scale1   # (D, D_ff)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, model_dim) * scale2   # (D_ff, D)
        self.b2 = np.zeros(model_dim)

        self.grad_W1 = np.zeros_like(self.W1)
        self.grad_b1 = np.zeros_like(self.b1)
        self.grad_W2 = np.zeros_like(self.W2)
        self.grad_b2 = np.zeros_like(self.b2)

    def forward(self, input_tensor: np.ndarray):
        # input_tensor: (B,T,D)
        hidden_pre_activation = input_tensor @ self.W1 + self.b1  # (B,T,D_ff)
        hidden_activation = np.maximum(0, hidden_pre_activation) # ReLU
        output_tensor = hidden_activation @ self.W2 + self.b2    # (B,T,D)
        self.cache = {
            "input_tensor": input_tensor,
            "hidden_pre_activation": hidden_pre_activation,
            "hidden_activation": hidden_activation
        }
        return output_tensor

    def backward(self, upstream_gradient: np.ndarray):
        # upstream_gradient: (B,T,D)
        cache = self.cache
        input_tensor = cache["input_tensor"]
        hidden_activation = cache["hidden_activation"]
        hidden_pre_activation = cache["hidden_pre_activation"]

        batch_size, sequence_length, model_dim = input_tensor.shape
        hidden_dim = hidden_pre_activation.shape[-1]

        # grads for W2 and b2
        grad_W2 = hidden_activation.reshape(batch_size*sequence_length, hidden_dim).T @ upstream_gradient.reshape(batch_size*sequence_length, model_dim)
        grad_b2 = np.sum(upstream_gradient, axis=(0,1))

        # backprop through W2
        grad_hidden_activation = upstream_gradient @ self.W2.T   # (B,T,hidden_dim)
        # backprop through ReLU
        grad_hidden_pre_activation = grad_hidden_activation * (hidden_pre_activation > 0).astype(float)

        # grads for W1 and b1
        grad_W1 = input_tensor.reshape(batch_size*sequence_length, model_dim).T @ grad_hidden_pre_activation.reshape(batch_size*sequence_length, hidden_dim)
        grad_b1 = np.sum(grad_hidden_pre_activation, axis=(0,1))

        # grad w.r.t input
        grad_input = grad_hidden_pre_activation @ self.W1.T      # (B,T,D)

        # store grads
        self.grad_W2 = grad_W2
        self.grad_b2 = grad_b2
        self.grad_W1 = grad_W1
        self.grad_b1 = grad_b1

        return grad_input


class TransformerBlock:
    def __init__(self, model_dim: int, num_heads: int, feedforward_multiplier: int = 4, random_seed: int = 0):
        self.attention = MultiHeadCausalSelfAttention(model_dim, num_heads, random_seed)
        hidden_dim = model_dim * feedforward_multiplier
        self.feedforward = PositionwiseFeedForward(model_dim, hidden_dim, random_seed + 1234)

    def forward(self, input_tensor: np.ndarray):
        # First sub-layer: attention
        # residual1 = X
        attention_output = self.attention.forward(input_tensor)     # (B,T,D)
        after_attention = input_tensor + attention_output           # residual
        normalized_after_attention, cache_norm1 = layer_norm_forward(after_attention)  # (B,T,D)

        # Second sub-layer: feedforward
        feedforward_output = self.feedforward.forward(normalized_after_attention)     # (B,T,D)
        after_feedforward = normalized_after_attention + feedforward_output          # residual
        normalized_after_feedforward, cache_norm2 = layer_norm_forward(after_feedforward)

        # store caches for backward
        self.cache = {
            "input_tensor": input_tensor,
            "after_attention": after_attention,
            "cache_norm1": cache_norm1,
            "normalized_after_attention": normalized_after_attention,
            "feedforward_output": feedforward_output,
            "after_feedforward": after_feedforward,
            "cache_norm2": cache_norm2
        }
        return normalized_after_feedforward

    def backward(self, upstream_gradient: np.ndarray):
        """
        upstream_gradient: dL/d(output_of_block) shape (B,T,D)
        returns dL/d(input_tensor) and accumulates gradients into submodules
        """
        cache = self.cache
        # Backprop through second LayerNorm:
        grad_after_feedforward = layer_norm_backward(upstream_gradient, cache["cache_norm2"])   # dL/d(after_feedforward)

        # Split residual: after_feedforward = normalized_after_attention + feedforward_output
        grad_feedforward_output = grad_after_feedforward     # contributes to feedforward module
        grad_normalized_after_attention_from_residual = grad_after_feedforward

        # Backprop through feedforward module
        grad_normalized_after_attention_from_ffn = self.feedforward.backward(grad_feedforward_output)  # returns dL/d(normalized_after_attention)

        # Total gradient w.r.t normalized_after_attention (sum from residual and ffn-backprop)
        grad_normalized_after_attention = grad_normalized_after_attention_from_residual + grad_normalized_after_attention_from_ffn

        # Backprop through first LayerNorm:
        grad_after_attention = layer_norm_backward(grad_normalized_after_attention, cache["cache_norm1"])  # dL/d(after_attention)

        # after_attention = input_tensor + attention_output
        grad_input_from_residual = grad_after_attention   # to input
        grad_attention_output = grad_after_attention     # to attention module

        # Backprop through attention module
        grad_input_from_attention = self.attention.backward(grad_attention_output)   # returns dL/d(input_tensor) from attention path

        # Total grad wrt input = grad_input_from_residual + grad_input_from_attention
        grad_input = grad_input_from_residual + grad_input_from_attention

        return grad_input

class MiniGPTFullBackprop:
    def __init__(self, vocab_size: int, sequence_length: int = 32, model_dim: int = 64,
                 num_layers: int = 2, num_heads: int = 4, random_seed: int = 0):
        np.random.seed(random_seed)
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.model_dim = model_dim

        # Embeddings
        self.token_embedding_matrix = np.random.randn(vocab_size, model_dim) * (1.0 / sqrt(vocab_size))
        self.position_embedding_matrix = np.random.randn(sequence_length, model_dim) * (1.0 / sqrt(sequence_length))

        # Output projection
        self.output_projection = np.random.randn(model_dim, vocab_size) * (1.0 / sqrt(model_dim))

        # Transformer stack
        self.transformer_blocks = [TransformerBlock(model_dim, num_heads, random_seed=random_seed + i*7) for i in range(num_layers)]

        # Grad accumulators
        self.grad_token_embedding = np.zeros_like(self.token_embedding_matrix)
        self.grad_position_embedding = np.zeros_like(self.position_embedding_matrix)
        self.grad_output_projection = np.zeros_like(self.output_projection)

    def forward(self, input_indices: np.ndarray):
        """
        input_indices: (batch, seq_len)
        returns: logits (batch, seq_len, vocab_size)
        and stores caches needed for backprop
        """
        batch_size, seq_len = input_indices.shape
        assert seq_len <= self.sequence_length

        # Embedding lookup
        token_embeds = self.token_embedding_matrix[input_indices]             # (B, T, D)
        position_indices = np.arange(seq_len)
        position_embeds = self.position_embedding_matrix[position_indices]    # (T, D)
        position_embeds_broadcast = position_embeds[None, :, :]               # (1, T, D)
        input_tensor = token_embeds + position_embeds_broadcast              # (B, T, D)

        # Pass through transformer blocks
        x = input_tensor
        for block in self.transformer_blocks:
            x = block.forward(x)

        # Final projection to logits
        logits = x @ self.output_projection   # (B, T, V)

        # Cache values for backward
        self.cache = {
            "input_indices": input_indices,
            "token_embeds": token_embeds,
            "position_embeds": position_embeds_broadcast,
            "pre_final_hidden": x
        }
        return logits

    def backward(self, logits: np.ndarray, target_indices: np.ndarray):
        """
        logits: (B, T, V)
        target_indices: (B, T)
        Computes grads wrt all parameters and returns nothing (grads stored in model).
        Uses standard cross-entropy loss L = -sum(log p_true) / (B*T)
        """
        batch_size, seq_len, vocab_size = logits.shape

        # Compute softmax probabilities and loss gradient
        logits_flat = logits.reshape(-1, vocab_size)      # (B*T, V)
        targets_flat = target_indices.reshape(-1)         # (B*T,)
        probs = stable_softmax(logits_flat, axis=1)       # (B*T, V)
        one_hot_targets = np.zeros_like(probs)
        one_hot_targets[np.arange(probs.shape[0]), targets_flat] = 1.0

        # Loss (for reporting) L = -1/(B*T) * sum log p_true
        log_probs_true = np.log(probs[np.arange(probs.shape[0]), targets_flat] + 1e-12)
        loss = -np.mean(log_probs_true)

        # Gradient of loss w.r.t logits: dL/dlogits = (probs - one_hot)/ (B*T)
        grad_logits = (probs - one_hot_targets).reshape(batch_size, seq_len, vocab_size) / (batch_size * seq_len)

        # Gradient w.r.t output_projection W_out: dW_out = H^T @ grad_logits where H is pre_final_hidden (B*T, D)
        pre_final_hidden = self.cache["pre_final_hidden"]   # (B, T, D)
        grad_output_projection = pre_final_hidden.reshape(batch_size*seq_len, self.model_dim).T @ grad_logits.reshape(batch_size*seq_len, vocab_size)

        # Gradient flowing into pre_final_hidden: dH = grad_logits @ W_out^T
        grad_pre_final_hidden = grad_logits @ self.output_projection.T   # (B, T, D)

        # Clear gradient accumulators
        # For transformer blocks, we'll backprop sequentially
        grad_hidden = grad_pre_final_hidden
        # Backprop through transformer blocks in reverse
        for block in reversed(self.transformer_blocks):
            grad_hidden = block.backward(grad_hidden)   # returns dL/d(input_of_block)

        # Now grad_hidden is gradient wrt input_tensor = token_embeds + position_embeds_broadcast (B,T,D)
        # So gradients for token embeddings and position embeddings are accumulations from token positions
        grad_token_embeddings_accum = np.zeros_like(self.token_embedding_matrix)   # (Vocab, D)
        grad_position_embeddings_accum = np.zeros_like(self.position_embedding_matrix) # (seq_len, D)

        input_indices = self.cache["input_indices"]    # (B, T)
        # accumulate grads
        B, T = input_indices.shape
        for b in range(B):
            for t in range(T):
                token_id = int(input_indices[b, t])
                grad_token_embeddings_accum[token_id] += grad_hidden[b, t]
                grad_position_embeddings_accum[t] += grad_hidden[b, t]

        # Store grads in model
        self.grad_token_embedding = grad_token_embeddings_accum
        self.grad_position_embedding = grad_position_embeddings_accum
        self.grad_output_projection = grad_output_projection

        return loss

    def step_sgd(self, learning_rate: float = 1e-2):
        # Apply SGD updates to parameters
        self.token_embedding_matrix -= learning_rate * self.grad_token_embedding
        self.position_embedding_matrix -= learning_rate * self.grad_position_embedding
        self.output_projection -= learning_rate * self.grad_output_projection

        # Also update parameters inside transformer blocks (attention and feedforward)
        for block in self.transformer_blocks:
            # Attention gradients
            attn = block.attention
            attn.W_query -= learning_rate * attn.gradient_W_query
            attn.W_key   -= learning_rate * attn.gradient_W_key
            attn.W_value -= learning_rate * attn.gradient_W_value
            attn.W_output-= learning_rate * attn.gradient_W_output

            # Feedforward gradients
            ffn = block.feedforward
            ffn.W1 -= learning_rate * ffn.grad_W1
            ffn.b1 -= learning_rate * ffn.grad_b1
            ffn.W2 -= learning_rate * ffn.grad_W2
            ffn.b2 -= learning_rate * ffn.grad_b2

# ---------------------------
# Training loop (mini-batch SGD)
# ---------------------------
def train_model_on_text(model: MiniGPTFullBackprop, token_indices: np.ndarray,
                        sequence_length: int = 32, batch_size: int = 8,
                        epochs: int = 5, learning_rate: float = 1e-2, verbose: bool = True):
    # Prepare sliding windows X (input) and Y (targets)
    total_chars = len(token_indices)
    max_start = total_chars - sequence_length - 1
    starts = np.arange(0, max_start, sequence_length)  # non-overlapping windows for simplicity
    num_sequences = len(starts)
    if num_sequences == 0:
        raise ValueError("Text too short for chosen sequence_length.")

    for epoch in range(1, epochs+1):
        np.random.shuffle(starts)
        epoch_loss = 0.0
        count = 0
        for batch_start in range(0, num_sequences, batch_size):
            batch_indices = starts[batch_start:batch_start+batch_size]
            actual_batch_size = len(batch_indices)
            if actual_batch_size == 0:
                continue

            X_batch = np.zeros((actual_batch_size, sequence_length), dtype=np.int32)
            Y_batch = np.zeros((actual_batch_size, sequence_length), dtype=np.int32)
            for i, start_idx in enumerate(batch_indices):
                X_batch[i] = token_indices[start_idx : start_idx + sequence_length]
                Y_batch[i] = token_indices[start_idx + 1 : start_idx + 1 + sequence_length]

            # Forward
            logits = model.forward(X_batch)

            # Backward (compute grads and loss)
            batch_loss = model.backward(logits, Y_batch)

            # Update parameters
            model.step_sgd(learning_rate)

            epoch_loss += batch_loss * actual_batch_size
            count += actual_batch_size

        avg_epoch_loss = epoch_loss / count
        if verbose:
            print(f"Epoch {epoch}/{epochs}  avg_loss={avg_epoch_loss:.6f}")

# ---------------------------
# Generation (autoregressive)
# ---------------------------
def generate_from_prompt(model: MiniGPTFullBackprop, tokenizer: CharacterTokenizer, prompt: str, max_new_tokens: int = 200, temperature: float = 1.0):
    # Encode prompt (clip last sequence_length tokens)
    prompt_indices = [tokenizer.char_to_index.get(ch, 0) for ch in prompt]
    prompt_indices = prompt_indices[-model.sequence_length:]
    generated = list(prompt_indices)
    for _ in range(max_new_tokens):
        context = np.array([generated[-model.sequence_length:]], dtype=np.int32) if len(generated) >= model.sequence_length else np.array([([0]*(model.sequence_length-len(generated)) + generated)], dtype=np.int32)
        logits = model.forward(context)   # (1, T, V)
        last_logits = logits[0, -1, :] / max(1e-9, temperature)
        probs = stable_softmax(last_logits)
        next_token = np.random.choice(model.vocab_size, p=probs)
        generated.append(int(next_token))
    return tokenizer.decode(np.array(generated, dtype=np.int32))

# ---------------------------
# Main: run everything
# ---------------------------
if __name__ == "__main__":
    txt_path = "training_text.txt"  # Provide your .txt file
    prompt_string = "My name is"

    # Read and tokenize data
    extracted_text = read_text_from_file(txt_path)

    print("Building tokenizer...")
    tokenizer = CharacterTokenizer(extracted_text)
    token_sequence = tokenizer.encode(extracted_text)
    print(f"Vocab size = {tokenizer.vocab_size}, total characters = {len(token_sequence)}")

    #hyperparameters
    SEQUENCE_LENGTH = 32
    MODEL_DIM = 64
    NUM_LAYERS = 2
    NUM_HEADS = 4
    BATCH_SIZE = 4
    EPOCHS = 1000
    LEARNING_RATE = 6e-3

    print("Initializing model...")
    model = MiniGPTFullBackprop(vocab_size=tokenizer.vocab_size,
                                sequence_length=SEQUENCE_LENGTH,
                                model_dim=MODEL_DIM,
                                num_layers=NUM_LAYERS,
                                num_heads=NUM_HEADS)

    print("Training (full backprop) — will be slow on CPU. Reduce sizes to speed up.")
    train_model_on_text(model, token_sequence,
                        sequence_length=SEQUENCE_LENGTH,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        learning_rate=LEARNING_RATE,
                        verbose=True)

    print("Generating text from prompt...")
    generated_text = generate_from_prompt(model, tokenizer, prompt_string, max_new_tokens=200, temperature=0.9)
    print("\n--- GENERATED ---\n")
    print(generated_text)
