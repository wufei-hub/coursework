import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

# =============== 1. 位置编码 ==================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=5000, dropout=0.1):
        super().__init__()
        pe = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return self.dropout(x)

# =============== 2. 缩放点积注意力 ==================
class ScaledDotProductAttention(nn.Module):
    def forward(self, q, k, v, mask=None):
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        return output, attn_weights

# =============== 3. 多头注意力 ==================
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        """
        q: (B, S_q, d_model)
        k: (B, S_k, d_model)
        v: (B, S_k, d_model)
        mask: 形状可广播到 (B, 1 或 H, S_q, S_k)
              - 编码器自注意力: (B,1,1,S_k)  -> 自动广播到 (B,H,S_q,S_k)（S_q=S_k）
              - 解码器自注意力: (B,1,S_q,S_q)
              - 编码器-解码器注意力: (B,1,1,S_k)
        """
        B, S_q, _ = q.shape
        S_k = k.size(1)
        H, d_k = self.num_heads, self.d_k

        # 1) 线性映射 + 拆头
        q = self.w_q(q).view(B, S_q, H, d_k).permute(0, 2, 1, 3).contiguous()  # (B,H,S_q,d_k)
        k = self.w_k(k).view(B, S_k, H, d_k).permute(0, 2, 1, 3).contiguous()  # (B,H,S_k,d_k)
        v = self.w_v(v).view(B, S_k, H, d_k).permute(0, 2, 1, 3).contiguous()  # (B,H,S_k,d_k)

        # 2) 点积注意力（scores 形状为 B,H,S_q,S_k）
        #    注意：不要再对 mask 额外 unsqueeze，让它依赖广播即可
        attn_output, attn_weights = self.attention(q, k, v, mask)  # (B,H,S_q,d_k)

        # 3) 合并头 -> (B,S_q,d_model)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(B, S_q, H * d_k)

        # 4) 输出线性
        output = self.dropout(self.w_o(attn_output))  # (B,S_q,d_model)
        return output, attn_weights



# =============== 4. 前馈网络 ==================
class PositionWiseFFN(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)     # (B, seq_len, d_ff)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)     # (B, seq_len, d_model)
        return x



# =============== 5. 残差 + 层归一化 ==================
class ResidualLayerNorm(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer_output):
        # 保证输入维度相同 (batch, seq_len, d_model)
        return self.norm(x + self.dropout(sublayer_output))


# =============== 6. 编码器块 ==================
class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionWiseFFN(d_model, d_ff, dropout)
        self.res1 = ResidualLayerNorm(d_model, dropout)
        self.res2 = ResidualLayerNorm(d_model, dropout)

    def forward(self, x, src_mask=None):
        attn_output, _ = self.self_attn(x, x, x, src_mask)  # (B, S, d_model)
        # 调试打印（只跑第一轮会看到）
        # print("self_attn out:", attn_output.shape, "x in:", x.shape)

        x = self.res1(x, attn_output)                      # (B, S, d_model)

        ffn_output = self.ffn(x)                           # (B, S, d_model)
        x = self.res2(x, ffn_output)                       # (B, S, d_model)
        return x



# =============== 7. 解码器块 ==================
class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.masked_self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.enc_dec_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionWiseFFN(d_model, d_ff, dropout)
        self.res1 = ResidualLayerNorm(d_model, dropout)
        self.res2 = ResidualLayerNorm(d_model, dropout)
        self.res3 = ResidualLayerNorm(d_model, dropout)

    def forward(self, x, enc_output, tgt_mask=None, src_mask=None):
        masked_attn_output, _ = self.masked_self_attn(x, x, x, tgt_mask)
        x = self.res1(x, masked_attn_output)
        enc_dec_attn_output, _ = self.enc_dec_attn(x, enc_output, enc_output, src_mask)
        # print("enc-dec attn out:", enc_dec_attn_output.shape)  # 期望 (B, tgt_len, d_model) = (16, 127, 512)
        x = self.res2(x, enc_dec_attn_output)
        ffn_output = self.ffn(x)
        return self.res3(x, ffn_output)

# =============== 8. Transformer 模型 ==================
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_layers=4, num_heads=8, d_ff=2048, max_seq_len=128, dropout=0.1):
        super().__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_seq_len, dropout)
        self.encoder = nn.ModuleList([EncoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder = nn.ModuleList([DecoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def encode(self, src, src_mask):
        x = self.pos_enc(self.dropout(self.src_emb(src)))
        # print("encode x shape:", x.shape)
        for layer in self.encoder:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt, enc_output, tgt_mask, src_mask):
        x = self.pos_enc(self.dropout(self.tgt_emb(tgt)))
        for layer in self.decoder:
            x = layer(x, enc_output, tgt_mask, src_mask)
        return x

    def forward(self, src, tgt, src_mask, tgt_mask):
        enc_output = self.encode(src, src_mask)
        dec_output = self.decode(tgt, enc_output, tgt_mask, src_mask)
        logits = self.fc_out(dec_output)
        return logits

# =============== 9. 掩码生成 ==================
def create_masks(src_ids, tgt_ids):
    src_mask = src_ids.ne(0).unsqueeze(1).unsqueeze(2)
    tgt_pad_mask = tgt_ids.ne(0).unsqueeze(1).unsqueeze(2)
    future_mask = torch.triu(torch.ones(1, tgt_ids.size(1), tgt_ids.size(1), device=src_ids.device), diagonal=1).eq(0)
    tgt_mask = tgt_pad_mask & future_mask
    return src_mask, tgt_mask

# =============== 10. 自定义Dataset ==================
class TranslationDataset(Dataset):
    def __init__(self, src_file, tgt_file, tokenizer, max_len=128):
        self.src_lines = open(src_file, encoding='utf-8').read().strip().split("\n")
        self.tgt_lines = open(tgt_file, encoding='utf-8').read().strip().split("\n")
        self.tokenizer = tokenizer
        self.max_len = max_len
        assert len(self.src_lines) == len(self.tgt_lines), "源语言和目标语言句子数量不一致"

    def __len__(self):
        return len(self.src_lines)

    def __getitem__(self, idx):
        src_text, tgt_text = self.src_lines[idx], self.tgt_lines[idx]
        src_enc = self.tokenizer(src_text, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt")
        tgt_enc = self.tokenizer(tgt_text, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt")
        src_ids = src_enc["input_ids"].squeeze(0)
        src_mask = src_enc["attention_mask"].squeeze(0)
        tgt_ids = tgt_enc["input_ids"].squeeze(0)[:-1]
        tgt_labels = tgt_enc["input_ids"].squeeze(0)[1:]
        return {"src_ids": src_ids, "src_mask": src_mask, "tgt_ids": tgt_ids, "tgt_labels": tgt_labels}

# =============== 11. 单轮训练函数 ==================
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch_idx, batch in enumerate(dataloader):
        src_ids = batch["src_ids"].to(device)
        src_mask = batch["src_mask"].to(device)
        tgt_ids = batch["tgt_ids"].to(device)
        tgt_labels = batch["tgt_labels"].to(device)

        src_mask, tgt_mask = create_masks(src_ids, tgt_ids)
        logits = model(src_ids, tgt_ids, src_mask, tgt_mask)

        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_labels.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

        # 👇 这里改：前3个batch都打、之后每100个batch打
        if batch_idx < 3 or (batch_idx + 1) % 100 == 0:
            avg_loss = total_loss / (batch_idx + 1)
            print(f"Batch {batch_idx+1}/{len(dataloader)} | Loss: {loss.item():.4f} | Avg: {avg_loss:.4f}", flush=True)

    return total_loss / len(dataloader)

# =============== 12. 主程序 ==================
def main():
    # 数据路径
    src_path = "/opt/data/private/yonghu/WF_NEW/homework/big_model/mid_term/datasets/train.clean.en.txt"
    tgt_path = "/opt/data/private/yonghu/WF_NEW/homework/big_model/mid_term/datasets/train.clean.de.txt"

    tokenizer = AutoTokenizer.from_pretrained("/opt/data/private/yonghu/WF_NEW/homework/big_model/mid_term/tokenizer")

    dataset = TranslationDataset(src_path, tgt_path, tokenizer, max_len=128)
    
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    for batch in train_loader:
        print("src_ids shape:", batch["src_ids"].shape)
        print("tgt_ids shape:", batch["tgt_ids"].shape)
        break   # 只打印第一个 batch 就够了
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Transformer(
        src_vocab_size=len(tokenizer),
        tgt_vocab_size=len(tokenizer),
        d_model=512,
        num_layers=4,
        num_heads=8,
        d_ff=2048,
        max_seq_len=128,
        dropout=0.1
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.95)

    num_epochs = 5
    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch+1}/{num_epochs} ===")
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        print(f"Epoch {epoch+1} Train Loss: {loss:.4f}")

    torch.save(model.state_dict(), "transformer_wmt17.pth")
    print("\n✅ 模型已保存到 transformer_wmt17.pth")

if __name__ == "__main__":
    main()
