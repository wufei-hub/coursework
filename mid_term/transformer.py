import os, csv, time, math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
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
        pe = pe.unsqueeze(0)  # (1, max_seq_len, d_model)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, S, d_model)
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return self.dropout(x)

# =============== 2. 缩放点积注意力 ==================
class ScaledDotProductAttention(nn.Module):
    def forward(self, q, k, v, mask=None):
        # q,k,v: (B, H, S_*, d_k)
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # (B,H,S_q,S_k)
        if mask is not None:
            # 约定 mask 为 bool，True 表示可见，False 表示屏蔽
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)  # (B,H,S_q,d_k)
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
        mask: 可广播到 (B, H, S_q, S_k)
              - 编码器自注意力: (B,1,1,S_k)
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

        # 2) 注意力
        attn_output, attn_weights = self.attention(q, k, v, mask)  # (B,H,S_q,d_k)

        # 3) 合并头 -> (B,S_q,d_model)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(B, S_q, H * d_k)

        # 4) 输出映射
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
        # x: (B, S, d_model)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# =============== 5. 残差 + 层归一化 ==================
class ResidualLayerNorm(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer_output):
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
        attn_output, _ = self.self_attn(x, x, x, src_mask)  # (B,S,d_model)
        x = self.res1(x, attn_output)
        ffn_output = self.ffn(x)
        x = self.res2(x, ffn_output)
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
        x = self.res2(x, enc_dec_attn_output)
        ffn_output = self.ffn(x)
        return self.res3(x, ffn_output)

# =============== 8. Transformer 模型 ==================
class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=512,
        num_layers=4,
        num_heads=8,
        d_ff=2048,
        max_seq_len=128,
        dropout=0.1,
        use_posenc=True,            # ← 新增：是否使用位置编码
    ):
        super().__init__()
        self.use_posenc = use_posenc  # ← 新增

        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_seq_len, dropout)  # 保留创建，方便随时打开
        self.encoder = nn.ModuleList([EncoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder = nn.ModuleList([DecoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def encode(self, src, src_mask):
        x = self.dropout(self.src_emb(src))
        if self.use_posenc:                 # ← 新增：按开关添加位置编码
            x = self.pos_enc(x)
        for layer in self.encoder:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt, enc_output, tgt_mask, src_mask):
        x = self.dropout(self.tgt_emb(tgt))
        if self.use_posenc:                 # ← 新增：按开关添加位置编码
            x = self.pos_enc(x)
        for layer in self.decoder:
            x = layer(x, enc_output, tgt_mask, src_mask)
        return x

    def forward(self, src, tgt, src_mask, tgt_mask):
        enc_output = self.encode(src, src_mask)
        dec_output = self.decode(tgt, enc_output, tgt_mask, src_mask)
        logits = self.fc_out(dec_output)  # (B, S_tgt-1, vocab)
        return logits


# =============== 9. 掩码生成 ==================
def create_masks(src_ids, tgt_ids):
    # bool 掩码：True=可见，False=屏蔽
    src_mask = src_ids.ne(0).unsqueeze(1).unsqueeze(2)  # (B,1,1,S_src)
    tgt_pad_mask = tgt_ids.ne(0).unsqueeze(1).unsqueeze(2)  # (B,1,1,S_tgt)
    future_mask = torch.triu(torch.ones(1, tgt_ids.size(1), tgt_ids.size(1), device=src_ids.device), diagonal=1).eq(0)  # (1,S_tgt,S_tgt) 下三角为 True
    tgt_mask = tgt_pad_mask & future_mask  # (B,1,S_tgt,S_tgt)
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
        src_enc = self.tokenizer(
            src_text, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt"
        )
        tgt_enc = self.tokenizer(
            tgt_text, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt"
        )
        src_ids = src_enc["input_ids"].squeeze(0)         # (S_src,)
        src_mask = src_enc["attention_mask"].squeeze(0)   # (S_src,)
        # Decoder 输入去掉最后一个 token；标签去掉第一个 token
        tgt_ids = tgt_enc["input_ids"].squeeze(0)[:-1]    # (S_tgt-1,)
        tgt_labels = tgt_enc["input_ids"].squeeze(0)[1:]  # (S_tgt-1,)
        return {"src_ids": src_ids, "src_mask": src_mask, "tgt_ids": tgt_ids, "tgt_labels": tgt_labels}

# =============== 日志工具 ===============
import os, csv, time

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def init_loggers(log_dir="logs", tag=None):
    """
    初始化训练与验证日志。
    参数:
      log_dir: 日志目录 (默认 logs/)
      tag: 实验标签 (如 "noPE", "baseline")，会加在文件名后方便区分。
    """
    ensure_dir(log_dir)

    # 根据 tag 决定文件名
    if tag:
        batch_log_path = os.path.join(log_dir, f"train_batches_{tag}.csv")
        epoch_log_path = os.path.join(log_dir, f"epochs_{tag}.csv")
    else:
        batch_log_path = os.path.join(log_dir, "train_batches.csv")
        epoch_log_path = os.path.join(log_dir, "epochs.csv")

    # 打开文件（追加模式）
    batch_f = open(batch_log_path, "a", newline="", encoding="utf-8")
    epoch_f = open(epoch_log_path, "a", newline="", encoding="utf-8")

    batch_writer = csv.writer(batch_f, delimiter=",")
    epoch_writer = csv.writer(epoch_f, delimiter=",")

    # 若文件为空，则写表头
    if os.stat(batch_log_path).st_size == 0:
        batch_writer.writerow(["timestamp","epoch","batch","num_batches","global_step","train_loss","avg_loss","lr"])
    if os.stat(epoch_log_path).st_size == 0:
        epoch_writer.writerow(["timestamp","epoch","train_loss","val_loss","val_ppl","best_val"])

    return (batch_f, epoch_f, batch_writer, epoch_writer)

def close_loggers(batch_f, epoch_f):
    try:
        batch_f.close()
    except:
        pass
    try:
        epoch_f.close()
    except:
        pass

def get_lr(optimizer):
    return optimizer.param_groups[0].get("lr", None)

# =============== 训练与验证（增强：逐 batch/epoch 记录） ===============
def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch_idx, global_step, batch_writer, log_every=100):
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    for batch_idx, batch in enumerate(dataloader):
        src_ids = batch["src_ids"].to(device)
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
        global_step += 1

        # —— 逐 batch 记日志 —— #
        if (batch_idx < 3) or ((batch_idx + 1) % log_every == 0) or (batch_idx + 1 == num_batches):
            avg_loss = total_loss / (batch_idx + 1)
            lr = get_lr(optimizer)
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            batch_writer.writerow([ts, epoch_idx, batch_idx+1, num_batches, global_step, f"{loss.item():.6f}", f"{avg_loss:.6f}", f"{lr:.8f}" if lr is not None else ""])
            # 打印到控制台
            print(f"Epoch {epoch_idx} | Batch {batch_idx+1}/{num_batches} | Loss: {loss.item():.4f} | Avg: {avg_loss:.4f} | LR: {lr:.2e}" if lr else
                  f"Epoch {epoch_idx} | Batch {batch_idx+1}/{num_batches} | Loss: {loss.item():.4f} | Avg: {avg_loss:.4f}", flush=True)

    return total_loss / num_batches, global_step

@torch.no_grad()
def evaluate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for batch in dataloader:
        src_ids = batch["src_ids"].to(device)
        tgt_ids = batch["tgt_ids"].to(device)
        tgt_labels = batch["tgt_labels"].to(device)

        src_mask, tgt_mask = create_masks(src_ids, tgt_ids)
        logits = model(src_ids, tgt_ids, src_mask, tgt_mask)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_labels.reshape(-1))
        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / max(1, n_batches)
    ppl = math.exp(avg_loss)
    return avg_loss, ppl

# =============== 主程序（增强：划分valid + 全过程落盘 + 中间权重） ===============
def main(args=None):
    # ------- 0) 实验标签 & 日志初始化（基于 no_posenc 自动命名） -------
    if args is None:
        class _A: pass
        args = _A()
        args.no_posenc = False
        args.tag = None
    tag = args.tag if getattr(args, "tag", None) else ("noPE" if args.no_posenc else "baseline")

    # 路径
    src_path = "/opt/data/private/yonghu/WF_NEW/homework/big_model/mid_term/datasets/train.clean.en.txt"
    tgt_path = "/opt/data/private/yonghu/WF_NEW/homework/big_model/mid_term/datasets/train.clean.de.txt"
    tok_path = "/opt/data/private/yonghu/WF_NEW/homework/big_model/mid_term/tokenizer"

    # 分词器
    tokenizer = AutoTokenizer.from_pretrained(tok_path)

    # 数据集与划分
    dataset = TranslationDataset(src_path, tgt_path, tokenizer, max_len=128)
    val_ratio = 0.05
    n_total = len(dataset)
    n_val = max(1, int(n_total * val_ratio))
    n_train = n_total - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=16, shuffle=False, num_workers=2, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # —— 日志与权重目录 —— #
    ensure_dir("logs"); ensure_dir("checkpoints")
    batch_f, epoch_f, batch_writer, epoch_writer = init_loggers(log_dir="logs", tag=tag)

    # 模型：这里根据 --no_posenc 决定是否使用位置编码
    model = Transformer(
        src_vocab_size=len(tokenizer),
        tgt_vocab_size=len(tokenizer),
        d_model=512, num_layers=4, num_heads=8, d_ff=2048, max_seq_len=128, dropout=0.1,
        use_posenc=(not args.no_posenc),
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, betas=(0.9, 0.98), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.95)

    best_val = float("inf")
    global_step = 0
    num_epochs = 5

    try:
        for epoch in range(1, num_epochs + 1):
            print(f"\n=== [{tag}] Epoch {epoch}/{num_epochs} ===")
            train_loss, global_step = train_one_epoch(
                model, train_loader, optimizer, criterion, device,
                epoch_idx=epoch, global_step=global_step, batch_writer=batch_writer, log_every=100
            )
            val_loss, val_ppl = evaluate_one_epoch(model, val_loader, criterion, device)
            scheduler.step()

            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            is_best = val_loss < best_val
            best_val = min(best_val, val_loss)
            epoch_writer.writerow([ts, epoch, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{val_ppl:.2f}", f"{best_val:.6f}"])
            print(f"[{tag}] Epoch {epoch} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | PPL: {val_ppl:.2f}")

            # 保存权重（带 tag 更清晰）
            torch.save(model.state_dict(), f"checkpoints/{tag}_epoch_{epoch}.pth")
            if is_best:
                torch.save(model.state_dict(), f"transformer_wmt17.best.{tag}.pth")
                print(f"🏆 Saved best -> transformer_wmt17.best.{tag}.pth")

        torch.save(model.state_dict(), f"transformer_wmt17.last.{tag}.pth")
        print(f"✅ 训练结束：best=transformer_wmt17.best.{tag}.pth, last=transformer_wmt17.last.{tag}.pth")
        print(f"📝 日志：logs/train_batches_{tag}.csv（逐 batch），logs/epochs_{tag}.csv（逐 epoch）")

    finally:
        close_loggers(batch_f, epoch_f)

# ====== 你已有的函数：create_masks / TranslationDataset / Transformer 等，保持不变 ======
# 确保把它们粘贴在本文件里或从你的模块导入



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_posenc", action="store_true", help="关闭位置编码（做消融）")
    parser.add_argument("--tag", type=str, default=None, help="实验标签（可选；不填则基于 no_posenc 自动生成）")
    args = parser.parse_args()

    # 统一在 main(args) 里完成：标签、日志、模型实例化
    main_args = args  # 传给 main
    main(main_args)

