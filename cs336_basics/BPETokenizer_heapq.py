from collections import defaultdict
from multiprocessing import Process, Queue
import regex as re
import os
import heapq
from typing import BinaryIO, List, Tuple, Dict

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def pre_tokenizer(text:str , special_tokens: list[str],drop_special_token: bool = True) -> list[str]:
    """
    Pre-tokenize the input text by splitting it based on special tokens.
    Returns a list of pre-tokens.
    我们对语料库进行预分词,这有助于我们统计字符对出现的频率。例如，单词 “text” 
    可能是一个出现了 10 次的预令牌。在这种情况下，当我们统计字符 “t” 和 “e” 相邻出现的频率时，
    会发现单词 “text” 中 “t” 和 “e” 是相邻的，我们可以将它们的计数增加 10，而无需遍历整个语料库。

    """
    parts = split_by_special_token(text,special_tokens)


    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    special_tokens_sorted = sorted(special_tokens, key=lambda x: -len(x))  # Sort by length descending

    tokens_list = []

    for part in parts:
        # 如果是特殊标记字符 
        if part in special_tokens_sorted : 
            if not drop_special_token:
                # 只记录一次即可 
                if part not in tokens_list:
                    tokens_list.append([part.encode("utf-8")])
        else:
            # 对非特殊标记字符进行分词,先分词再编码
            str_tokens = re.findall(PAT, part)
            part_tokens = [token.encode("utf-8") for token in str_tokens]
            tokens_list.append(part_tokens)
            

    tokens = [token for part_tokens in tokens_list for token in part_tokens]
    # [b'Hello', b' world', b'!', b' ', b'<|endoftext|>', b' Great', b'!']
    return tokens
     
 
def split_by_special_token(text:str ,special_tokens:list[str]) -> list[str]:
    """
    Split on the special tokens
    example: 
        text = "Hello world! <|endoftext|> Great!" 
        special_tokens = "<|endoftext|>"
        result = ['Hello world! ', '<|endoftext|>', ' Great!']
    """
    parts = [text]
    special_tokens_sorted = sorted(special_tokens,key=lambda x:-len(x))
    # 如果没有特殊标记符号，直接返回原文即可 
    if not special_tokens_sorted:
        return parts
    # 否则就要切分 
    else: 
        # 加上转义符号 
        pattern = "|".join(re.escape(special_tok) for special_tok in special_tokens_sorted)
        # print("pattern:",pattern)
        parts = re.split('(' + pattern + ')', text)
    return parts
def worker(text:str , special_tokens: list[str],q: Queue) : 
    pretokens = pre_tokenizer(text,special_tokens)
    q.put(pretokens)
    # print("========done========")


# ------------------ 可更新优先队列 ------------------
class UpdatablePriorityQueue:
    def __init__(self):
        self.heap = []
        self.entry_finder = {}  # pair -> freq

    def push(self, pair, freq):
        entry = (-freq, pair)
        self.entry_finder[pair] = freq
        heapq.heappush(self.heap, entry)

    def pop(self):
        while self.heap:
            neg_freq, pair = heapq.heappop(self.heap)
            if pair in self.entry_finder and self.entry_finder[pair] == -neg_freq:
                del self.entry_finder[pair]
                return pair, -neg_freq
        raise KeyError("pop from empty priority queue")

    def update(self, pair, freq):
        self.push(pair, freq)

    def empty(self):
        return len(self.entry_finder) == 0

# ------------------ merge 函数 ------------------
def merge(counts, index_dict, pretokens, max_pair, new_index, pq: UpdatablePriorityQueue):
    # 获取所有位置 (序列索引, 位置)
    positions = index_dict[max_pair]
    index_dict[max_pair] = set()  # 清空原来的位置记录

    for seq_idx, pos in positions:
        token_seq = pretokens[seq_idx]
        # 直接在序列中替换
        if token_seq[pos] == max_pair[0] and token_seq[pos + 1] == max_pair[1]:
            token_seq[pos] = new_index
            del token_seq[pos + 1]
        pretokens[seq_idx] = token_seq

    # 更新 counts 和 index_dict，只处理受影响的 pair
    new_positions = set()
    for seq_idx, pos in positions:
        token_seq = pretokens[seq_idx]
        # 左邻居
        if pos > 0:
            left_pair = (token_seq[pos - 1], new_index)
            counts[left_pair] += 1
            index_dict[left_pair].add((seq_idx, pos - 1))
            pq.update(left_pair, counts[left_pair])
        # 右邻居
        if pos < len(token_seq) - 1:
            right_pair = (new_index, token_seq[pos + 1])
            counts[right_pair] += 1
            index_dict[right_pair].add((seq_idx, pos))
            pq.update(right_pair, counts[right_pair])

# ------------------ train_bpe ------------------
def train_bpe(input_path: str | os.PathLike,
              vocab_size: int,
              special_tokens: list[str],
              **kwargs) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    num_merges = max(vocab_size - len(special_tokens) - 256, 0)
    vocab = {x: bytes([x]) for x in range(256)}
    for i, token in enumerate(special_tokens):
        vocab[256 + i] = token.encode("utf-8")

    merges = []

    # ------------------ 读取文件并分块 ------------------
    num_processes = 4
    chunk_list = []
    with open(input_path, "rb") as f:
        
        boundaries = find_chunk_boundaries(f, num_processes, "<|endoftext|>".encode("utf-8"))
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunk = chunk.replace("\r\n", "\n").replace("\r", "\n")
            chunk_list.append(chunk)

    # ------------------ 多进程预分词 ------------------
 
    q = Queue()
    process = []
    for chunk in chunk_list:
        p = Process(target=worker, args=(chunk, special_tokens,q))
        p.start()
        process.append(p)
    pre_tokens_list = [q.get() for _ in process]
    for p in process:
        p.join()
    pretokens = [token for tokens in pre_tokens_list for token in tokens]

    # ------------------ 初始化 counts 和 index_dict ------------------
    counts = defaultdict(int)
    index_dict = defaultdict(set)
    pq = UpdatablePriorityQueue()

    for seq_idx, token_seq in enumerate(pretokens):
        for pos in range(len(token_seq) - 1):
            pair = (token_seq[pos], token_seq[pos + 1])
            counts[pair] += 1
            index_dict[pair].add((seq_idx, pos))

    # 构建优先队列
    for pair, freq in counts.items():
        pq.push(pair, freq)

    # ------------------ 迭代合并 ------------------
    for i in range(num_merges):
        if pq.empty():
            break
        max_pair, max_freq = pq.pop()
        new_index = 256 + len(special_tokens) + i
        index1, index2 = max_pair
        vocab[new_index] = vocab[index1] + vocab[index2]
        merges.append((vocab[index1], vocab[index2]))

        merge(counts, index_dict, pretokens, max_pair, new_index, pq)

    return vocab, merges
if __name__ == "__main__":
    # example_of_bytes_wrong()
    # res = split_by_special_token("Hello world! <|endoftext|> Great!",["<|endoftext|>"])
    # print(res)
    # tokens = pre_tokenizer("Hello world! <|endoftext|> Great!",["<|endoftext|>"],drop_special_token=False)


    # print(tokens)
    input_path = r'tests\fixtures\tinystories_sample.txt'
    train_bpe(input_path=input_path,vocab_size=1000,special_tokens=["<|endoftext|>"])


