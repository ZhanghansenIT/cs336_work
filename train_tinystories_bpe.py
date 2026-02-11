from pathlib import Path
from tests.adapters import run_train_bpe  


# 验证集
DATA_PATH = Path("data/TinyStories-valid.txt")  
VOCAB_SIZE = 1000
SPECIAL_TOKENS = ["<|endoftext|>"]

def main():
    assert DATA_PATH.exists(), f"{DATA_PATH} 不存在"

    vocab, merges = run_train_bpe(
        input_path=DATA_PATH,
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
    )

    # 简单存成文件，后面构建 tokenizer / 训练模型会用到
    import json

    # vocab: id -> bytes，需要转成可 json 的
    vocab_json = {int(k): v.decode("utf-8", errors="replace") for k, v in vocab.items()}
    with open("tinystories_vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab_json, f, ensure_ascii=False)

    with open("tinystories_merges.txt", "w", encoding="utf-8") as f:
        for a, b in merges:
            f.write(a.decode("utf-8", errors="replace") + " " + b.decode("utf-8", errors="replace") + "\n")

    print("训练完成，已保存 tinystories_vocab.json / tinystories_merges.txt")

if __name__ == "__main__":
    main()