import streamlit as st
import torch
import torch.nn as nn
import math
import json


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: [1, max_len, d_model]
        self.register_buffer("pe", pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)
        q = q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        output = self.out_linear(output)
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(0.1)
    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=2048):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
    def forward(self, x, mask=None):
        attn_output = self.attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout2(ff_output))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=2048):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.enc_dec_attn = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        self_attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(self_attn_output))
        enc_dec_attn_output = self.enc_dec_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout2(enc_dec_attn_output))
        ff_output = self.ff(x)
        x = self.norm3(x + self.dropout3(ff_output))
        return x

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_len=5000):
        super(Encoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.dropout = nn.Dropout(0.1)
    def forward(self, src, mask=None):
        x = self.embed(src) * math.sqrt(self.embed.embedding_dim)
        x = self.pos_enc(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_len=5000):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.dropout = nn.Dropout(0.1)
        self.fc_out = nn.Linear(d_model, vocab_size)
    def forward(self, tgt, enc_output, src_mask=None, tgt_mask=None):
        x = self.embed(tgt) * math.sqrt(self.embed.embedding_dim)
        x = self.pos_enc(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        output = self.fc_out(x)
        return output

class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_layers=3, num_heads=8, d_ff=2048, max_len=5000):
        super(TransformerModel, self).__init__()
        self.encoder = Encoder(src_vocab_size, d_model, num_layers, num_heads, d_ff, max_len)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_layers, num_heads, d_ff, max_len)
    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        enc_output = self.encoder(src, src_mask)
        output = self.decoder(tgt, enc_output, src_mask, tgt_mask)
        return output


@st.cache_resource
def load_vocab():
    with open("src_vocab.json", "r") as f:
        src_vocab = json.load(f)
    with open("tgt_vocab.json", "r") as f:
        tgt_vocab = json.load(f)
    return src_vocab, tgt_vocab

src_vocab, tgt_vocab = load_vocab()

@st.cache_resource
def load_model():
    device = torch.device("cpu")
    model = TransformerModel(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=512,
        num_layers=3,
        num_heads=8,
        d_ff=2048
    )
    checkpoint = torch.load("transformer_model_epoch_1.pth", map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    return model

model = load_model()


def tokenize(text, vocab):
    text = str(text)
    tokens = text.strip().split()
    token_ids = [vocab.get(token, vocab["<unk>"]) for token in tokens]
    return token_ids

def tokenize_src(text):
    return tokenize(text, src_vocab)

def translate(model, src_sentence, max_len=100):
    device = torch.device("cpu")
    model.eval()
    # Tokenize and create tensor
    src_tokens = tokenize_src(src_sentence)
    src_tensor = torch.tensor(src_tokens).unsqueeze(0).to(device)
    with torch.no_grad():
        enc_output = model.encoder(src_tensor)
    # Begin decoding with <sos>
    tgt_indices = [tgt_vocab["<sos>"]]
    for _ in range(max_len):
        tgt_tensor = torch.tensor(tgt_indices).unsqueeze(0).to(device)
        tgt_mask = model.generate_square_subsequent_mask(tgt_tensor.size(1)).to(device)
        with torch.no_grad():
            output = model.decoder(tgt_tensor, enc_output, tgt_mask=tgt_mask)
        next_token = output.argmax(dim=-1)[:, -1].item()
        tgt_indices.append(next_token)
        if next_token == tgt_vocab["<eos>"]:
            break
    # Build a reverse mapping for the target vocabulary
    rev_tgt_vocab = {idx: token for token, idx in tgt_vocab.items()}
    translated_tokens = [rev_tgt_vocab.get(idx, "<unk>") for idx in tgt_indices]
    # Remove special tokens for clarity
    result = " ".join(token for token in translated_tokens if token not in ["<sos>", "<eos>"])
    return result


st.title("Pseudocode to C++ Translator")
st.markdown("Enter your pseudocode below and click **Translate** to see the generated C++ code.")

user_input = st.text_area("Pseudocode Input", height=150)

if st.button("Translate"):
    if user_input.strip():
        translation = translate(model, user_input)
        st.subheader("Translated C++ Code:")
        st.code(translation, language="cpp")
    else:
        st.error("Please enter some pseudocode.")
