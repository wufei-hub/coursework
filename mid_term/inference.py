import torch
from transformers import AutoTokenizer
from transformer import Transformer, create_masks  # 导入你自己定义的模块

def translate_sentence(model, tokenizer, src_text, device, max_len=128):
    model.eval()
    
    # 1️⃣ 对德语输入分词
    src_enc = tokenizer(src_text, return_tensors="pt", max_length=max_len, truncation=True, padding="max_length")
    src_ids = src_enc["input_ids"].to(device)
    src_mask = src_enc["attention_mask"].unsqueeze(1).unsqueeze(2).to(device)

    # 2️⃣ 初始化目标序列（decoder输入）：只放一个 <pad> 或 <bos> token
    tgt_ids = torch.tensor([[tokenizer.pad_token_id]], dtype=torch.long).to(device)

    # 3️⃣ Greedy decoding：一步步生成德语
    for _ in range(max_len):
        src_mask, tgt_mask = create_masks(src_ids, tgt_ids)
        with torch.no_grad():
            logits = model(src_ids, tgt_ids, src_mask, tgt_mask)
        
        next_token_logits = logits[:, -1, :]  # 最后一个时间步
        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)  # Greedy
        tgt_ids = torch.cat([tgt_ids, next_token_id], dim=1)

        # 如果生成了 <eos> 就停
        if next_token_id.item() == tokenizer.eos_token_id:
            break

    # 4️⃣ 解码为字符串
    output_text = tokenizer.decode(tgt_ids.squeeze(), skip_special_tokens=True)
    return output_text

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "/opt/data/private/yonghu/WF_NEW/homework/big_model/mid_term/transformer_wmt17.best.pth"
    tokenizer_path = "/opt/data/private/yonghu/WF_NEW/homework/big_model/mid_term/tokenizer"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # 5️⃣ 创建模型结构并加载权重
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

    model.load_state_dict(torch.load(model_path, map_location=device))
    print("✅ 模型加载成功！")

    # 6️⃣ 输入德语句子
    while True:
        src_text = input("\n请输入英语句子（或输入 quit 退出）：\n> ")
        if src_text.lower() == "quit":
            break
        translation = translate_sentence(model, tokenizer, src_text, device)
        print("结果：", translation)

if __name__ == "__main__":
    main()
