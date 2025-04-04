# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

np.random.seed(27)

data = pd.read_csv("./data/data_h7n9_severity.csv")

df = data.sample(frac=1).reset_index(drop=True) #shuffle data

def split_into_chunks(total, num_chunks, min_val, max_val):
    while True:
        # Generate (num_chunks - 1) random numbers between min_val and max_val
        chunks = []
        remaining = total
        
        # Generate chunks ensuring they stay within [min_val, max_val]
        for _ in range(num_chunks - 1):
            # The maximum possible value for the next chunk to leave room for remaining chunks
            upper = min(max_val, remaining - (num_chunks - len(chunks) - 1) * min_val)
            lower = max(min_val, remaining - (num_chunks - len(chunks) - 1) * max_val)
            
            if lower > upper:  # No valid chunk possible, retry
                break
            
            chunk = np.random.randint(lower, upper)
            chunks.append(chunk)
            remaining -= chunk
        
        else:  # Only if the loop completes without breaking
            # The last chunk is whatever remains
            last_chunk = remaining
            if min_val <= last_chunk <= max_val:
                chunks.append(last_chunk)
                return chunks
        
        # If we're here, the conditions weren't met, so we retry

total = 395
num_chunks = 9
min_chunk = 18
max_chunk = 50

chunks = split_into_chunks(total, num_chunks, min_chunk, max_chunk)

np.random.shuffle(chunks)

for c in range(len(chunks)):
    if c < 9:
        site = "S0"+str(c+1)
    else:
        site = "S"+str(c+1)
    d = df[:chunks[c]]
    d.to_csv("chunked_data/df_"+str(site)+".csv", index=False)
    idx = d.index
    df = df.drop(idx)