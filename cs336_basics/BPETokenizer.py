

from .pretokenization_example import find_chunk_boundaries
import regex as re 
from typing import List, Tuple, Dict
import os 
from multiprocessing import Process, Queue
from collections import defaultdict
import heapq 

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


def train_bpe(input_path: str | os.PathLike,
              vocab_size: int,
              special_tokens: list[str],
              **kwargs)->tuple[dict[int,bytes],list[tuple[bytes, bytes]]]:
    if len(special_tokens) ==0 : 
        special_tokens = []
    # 不能为负数 
    num_merges = max(vocab_size -len(special_tokens) -256,0) 

    vocab = {} 
    vocab = {x:bytes([x]) for x in range(0,256)}

    # 加上特殊字符 
    for i , token in enumerate(special_tokens) : 
        vocab[256+i] = token.encode("utf-8")
    merges = []

    # Chunk the text file
    num_processes = 4
    chunk_list = []
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, "<|endoftext|>".encode("utf-8"))

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunk = chunk.replace("\r\n", "\n").replace("\r","\n")  # Normalize line endings
            chunk_list.append(chunk)
    
    # print(f"chunk number :  {len(chunk_list)}")
    # for chunk in chunk_list: 
    #     print(chunk)
    #     print

    pre_tokens_list = []
    process = []
    q = Queue()

    for chunk in chunk_list:
        # 对于分割的每个chunk 都有一个进程去处理 
        p = Process(target=worker, args=(chunk, special_tokens,q))
        p.start()
        process.append(p)

    pre_tokens_list = [q.get() for _ in process]
    for p in process:
        p.join()
    pretokens = [token for tokens in pre_tokens_list for token in tokens]
    # print(f"pretokens number : {len(pretokens)}")
    # print(f"pretokens : {pretokens}")


    counts = defaultdict(int)
    index_dict = defaultdict(set)

    for j , pretoken in enumerate(pretokens):

        # pretoken  :  b'Hello'
        # print(f"pretoken: {pretoken}") 
        for index1 , index2 in zip(pretoken,pretoken[1:]) : 
            
            # print(f"index1 : {index1}-{pretoken} , index2 : {index2}-{pretoken[1:]} ")
            # 统计两个相邻的字母 出现次数
            counts[index1, index2] += 1
            # 记录这个相邻pair出现在哪些预分词中
            index_dict[index1,index2].add(j)
    
    # print(f"counts: {counts}")

    for i in range(num_merges):  
        max_pair = max(
            counts.items(),  # 待排序的字典项，形式为 ((index1, index2), count)
            key=lambda x: (  # 排序键的规则
                x[1],  # 频率（主要排序依据）
                vocab[x[0][0]].decode("utf-8", errors="ignore"),  # 第一个字符的字符串形式
                vocab[x[0][1]].decode("utf-8", errors="ignore")   # 第二个字符的字符串形式
            )
        )[0]  # 取最终结果的键（即 (index1, index2)）
        index1 ,index2 = max_pair 
        # print(f"max_pair: {max_pair}, freq: {counts[max_pair]}")
        # print(index_dict[max_pair])
        # print(chr(index1), chr(index2))
        new_index = 256 + len(special_tokens) + i

        vocab[new_index] = vocab[index1] + vocab[index2]
      
        merges.append((vocab[index1], vocab[index2]))
    
        # print(max_pair)
        # print(f"new_index: {new_index}, vocab[new_index]: {vocab[new_index]}")
        #   break 
        merge(counts, index_dict, pretokens, max_pair, new_index)
        # break 
    return (vocab, merges)



def merge(counts: dict[tuple[int, int], int], index_dict: dict[tuple[int, int],set[int]], pretokens: list[list[int]], max_pair: (int, int), new_index: int):

    """Merge the pairs with highest frequency and update counts, index_dict"""

    # 哪些token序列中出现这些频繁出现的 pairs 
    index_set = index_dict[max_pair]
    # print(f"index_set: {index_set}")
    # 遍历进行替换
    for i in index_set: 
        pretoken = pretokens[i]
        # print(f"pretoken: {pretoken}")
        new_pretoken = []

        pos_list = [] 
        pos = 0 

        j = 0
        while j < len(pretoken):
        
            # 如果是频繁出现的字符对 
            if (j<len(pretoken)-1 and (pretoken[j], pretoken[j+1]) == max_pair):
                new_pretoken.append(new_index)
                pos_list.append(pos)
                j += 2  # 跳过这两个字符,只把新的字符加入新预分词
            else: 
                new_pretoken.append(pretoken[j])
                
                j += 1
            pos += 1
        
        # 更新 counts and index_dict 
        for pos in pos_list :  
            counts[max_pair] -= 1


            # 这里要处理原来没替换之前的字符与前一个字符和后一个字符的count 计算
            if pos > 0 : 
                # 如果不是第一个字符，才有前一个字符
                if new_pretoken[pos-1] == new_index : 
                    
                    counts[(max_pair[1],max_pair[0])] -= 1
                else:
                    # 与前一个字符的计数 
                    counts[(new_pretoken[pos-1], max_pair[0])] -= 1

                counts[(new_pretoken[pos-1], new_index)] += 1
                index_dict[(new_pretoken[pos-1], new_index)].add(i)

            if pos < len(new_pretoken) - 1:
                if new_pretoken[pos+1] == new_index :
                    counts[(max_pair[1], max_pair[0])] -= 1
                else:
                    # 与后一个字符的计数 
                    counts[(max_pair[1], new_pretoken[pos+1])] -= 1

                counts[(new_index, new_pretoken[pos+1])] += 1
                index_dict[(new_index, new_pretoken[pos+1])].add(i) 


        pretokens[i] = new_pretoken







def example_of_bytes_wrong() : 
    """
    在用 decode("utf-8") 解码单个字节，而这个字节本身不足以组成一个完整的 UTF-8 字符。
    UTF-8 对于汉字会占用多个字节，所以必须把它们拼在一起才能解码。"""
    # "你好"  -> b'\xe4\xbd\xa0\xe5\xa5\xbd'
    string_input = "你好".encode("utf-8")
    print(string_input)
    print(bytes([string_input[0]]))
    print(len(bytes([string_input[0]])))
    res = " ".join([bytes([b]).decode("utf-8") for b in string_input])
    print(res)    


if __name__ == "__main__":
    # example_of_bytes_wrong()
    # res = split_by_special_token("Hello world! <|endoftext|> Great!",["<|endoftext|>"])
    # print(res)
    # tokens = pre_tokenizer("Hello world! <|endoftext|> Great!",["<|endoftext|>"],drop_special_token=False)


    # print(tokens)
    input_path = r'tests\fixtures\tinystories_sample.txt'
    train_bpe(input_path=input_path,vocab_size=1000,special_tokens=["<|endoftext|>"])









