#!/usr/bin/env python
# coding: utf-8

# # <u> Dataset Creation 

# In[2]:


def retrieve_character_dialogues(character,file):
    dialogues = []
    with open(file) as f:
        for line in f:
            search_line = line.lower()
            anya_found = search_line.find(character)
            comma_found = search_line.find(',,')
            if anya_found != -1 and comma_found != -1 and anya_found < comma_found:
                dialogues.append(line[comma_found+2:])
    return dialogues


# ## after uploading all .ass files and folders to the current working directory

# In[3]:


from pathlib import Path
import re


# In[4]:


bracket_content = re.compile(r'\{.*?\}|shock!|\n')


# In[5]:


p = Path('.')
paths = list(p.glob('**/*.eng.ass'))
character = 'anya'
dialogues = []
for path in paths:
    dialogues.extend(retrieve_character_dialogues(character,path))
cleaned_dialogues = bracket_content.sub("","".join(dialogues))


# ## Preliminary Checks

# In[ ]:


import sys

size_kb = sys.getsizeof(cleaned_dialogues) // 1024
print(f"Memory size: {size_kb} KB")

print(f"Number of dialogues: {len(cleaned_dialogues)}")

chars = sorted(set(cleaned_dialogues))
vocab_size = len(chars)
print(f"Vocabulary size: {vocab_size}")
print(f"Possible Vocabulary = [{''.join(chars)}]")


# > ### <u> NOTE: </u>
# >
# > - <b>The amount of data is quite low but we dont really care about overfitting for this model right now, although the we still  should have around an MB of 
# > data if possible
# >   
# >
# > <br>

# ## Encoder and Decoder Functions

# In[7]:


stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s : [stoi[c] for c in s]
decode = lambda l : ''.join(itos[i] for i in l)


# In[8]:


encoded_text = encode("Hehe")
print('encoded text : ', encoded_text)
decoded_text = decode(encoded_text)
print('decoded text :',decoded_text)


# ## Storing data

# In[9]:


import torch 
data = torch.tensor(encode(cleaned_dialogues),dtype=torch.long)
print(data.shape, data.dtype)


# ## Splitting data

# In[10]:


n = int(0.8*len(data))
train_data = data[:n]
val_data = data [n:]


# ## visualizing one training example 

# In[11]:


blocksize = 8 
x = train_data[:blocksize]
y = train_data[1:blocksize+1]
for t in range(blocksize):
    context = x[:t+1]
    target = y[t]
    print(f'given context {context} the next predicted token should be {target}')


# a sequence of n characters typically has n-1 training examples, we take (n+1) characters  to match the 'block_size' or number of training examples per sequence. eg: for block_size 8 we would consider a sequence of 9 characters so we get 8 training examples

# ## Creating Data Loader

# In[ ]:


#setting seed for consistent output
torch.manual_seed(74)
batch_size = 4 
block_size = 8 
# gets 4 sequences of 8 characters
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size,(batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1 : i+block_size+1] for i in ix])
    return x,y


# In[13]:


xb, yb = get_batch('train')

print("Inputs (xb):")
print(f"  Shape: {xb.shape}, \n Values: {xb}")
print("\nTargets (yb):")
print(f"  Shape: {yb.shape}, \n Values: {yb}")


# In[23]:


for b in range(batch_size):
    for t in range(block_size):
        context = xb[b, :t+1]   # Show context slice up to t+1
        target = yb[b, t]       # Show current target
        print(f"Batch: {b}, Time Step: {t+1}")
        print(f"Context: {context}")
        print(f"Target: {target}")
        print("_" * 40)         


# # SIMPLEST BASELINE

# ## Defining the Bigram

# In[ ]:


import torch 
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

torch.manual_seed(74)
#TL:DR each row maps to a character and gives logits or it, i.e next character predictions given an input characters idx
class BigramLanguageModel(nn.Module):

    def __init__(self,vocab_size):
        super().__init__()
        # the trainable params here are just weights of size vocab_size * vocab_size
        self.token_embedding_table = nn.Embedding(vocab_size,vocab_size)

    def forward(self, idx, targets=None):

        logits = self.token_embedding_table(idx)
        if targets is None:
            loss = None
        else:
            logits = rearrange(logits, 'b t c -> (b t) c')
            targets = rearrange(targets, 'b t -> (b t)' )
            loss = F.cross_entropy(logits,targets)

        return logits,loss
    def generate(self,idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits,loss = self(idx)

            logits = logits[:, -1 , :]

            probs = F.softmax(logits,dim=-1)
            idx_next =torch.multinomial(probs,num_samples=1)
            idx = torch.cat((idx,idx_next),dim=1)
        return idx
m = BigramLanguageModel(vocab_size)
logits,loss = m(xb,yb)
print(loss)
print(decode(m.generate(idx = torch.zeros((1,1),dtype=torch.long), max_new_tokens=100)[0].tolist()))


# > **Note**  
# > Even though we pass the entire sequence of logits into the next forward pass, this is still a **bigram model** — it only uses the **immediately previous character** to predict the next one.  
# >  
# > The reason we concatenate the entire sequence and then take only the last prediction is for **consistency**. The same forward function can then be reused for more complex models (like Transformers), where the full sequence context actually matters.
# >
# > also to use the model on gpu both the data and the parameters need to be offloaded to the gpu to work

# In[16]:


optimizer = torch.optim.AdamW(m.parameters(), lr = 1e-2)


# In[33]:


batch_size = 32
for step in range(10000):

    xb,yb = get_batch('train')

    logits , loss = m(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
print("loss : ",loss.item())


# In[43]:


print(decode(m.generate(idx = torch.zeros((1,1),dtype=torch.long), max_new_tokens=50)[0].tolist()))

