"""
任务五：Seq2seq德英翻译（简化版）
使用双层LSTM实现编码器-解码器架构
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
from tqdm import tqdm
import pickle
import os

# 设置随机种子
SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# ==================== 数据准备 ====================

class SimpleVocab:
    """简单的词汇表"""
    def __init__(self):
        self.word2idx = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
        self.idx2word = {0: '<pad>', 1: '<sos>', 2: '<eos>', 3: '<unk>'}
        self.word_count = 4
    
    def add_sentence(self, sentence):
        for word in sentence.split():
            if word not in self.word2idx:
                self.word2idx[word] = self.word_count
                self.idx2word[self.word_count] = word
                self.word_count += 1
    
    def sentence_to_indices(self, sentence, max_len=50):
        indices = [self.word2idx.get(word, 3) for word in sentence.split()]
        indices = [1] + indices + [2]  # 添加<sos>和<eos>
        if len(indices) < max_len:
            indices += [0] * (max_len - len(indices))
        else:
            indices = indices[:max_len-1] + [2]
        return indices

# 创建简单的德英翻译数据集
train_data = [
    ("Guten Morgen", "Good morning"),
    ("Wie geht es dir", "How are you"),
    ("Ich liebe dich", "I love you"),
    ("Danke schön", "Thank you"),
    ("Auf Wiedersehen", "Goodbye"),
    ("Ich bin müde", "I am tired"),
    ("Das ist gut", "That is good"),
    ("Ich verstehe nicht", "I do not understand"),
    ("Wo ist das", "Where is that"),
    ("Ich habe Hunger", "I am hungry"),
    ("Es ist kalt", "It is cold"),
    ("Ich bin glücklich", "I am happy"),
    ("Das ist schön", "That is beautiful"),
    ("Ich komme aus Deutschland", "I come from Germany"),
    ("Wie heißt du", "What is your name"),
    ("Ich spreche Deutsch", "I speak German"),
    ("Das Wetter ist schön", "The weather is nice"),
    ("Ich gehe nach Hause", "I go home"),
    ("Bis später", "See you later"),
    ("Gute Nacht", "Good night"),
]

print(f"\n数据集大小: {len(train_data)} 个句子对")

# 构建词汇表
print("\n构建词汇表...")
src_vocab = SimpleVocab()
tgt_vocab = SimpleVocab()

for src, tgt in train_data:
    src_vocab.add_sentence(src)
    tgt_vocab.add_sentence(tgt)

print(f"德语词汇量: {src_vocab.word_count}")
print(f"英语词汇量: {tgt_vocab.word_count}")

# ==================== 数据集类 ====================

class TranslationDataset(Dataset):
    def __init__(self, data, src_vocab, tgt_vocab):
        self.data = data
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        src, tgt = self.data[idx]
        src_indices = torch.tensor(self.src_vocab.sentence_to_indices(src))
        tgt_indices = torch.tensor(self.tgt_vocab.sentence_to_indices(tgt))
        return src_indices, tgt_indices

# 创建数据加载器
dataset = TranslationDataset(train_data, src_vocab, tgt_vocab)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# ==================== 模型定义 ====================

class Encoder(nn.Module):
    """编码器：双层LSTM"""
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell

class Decoder(nn.Module):
    """解码器：双层LSTM"""
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input, hidden, cell):
        input = input.unsqueeze(1)
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    """Seq2seq模型"""
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    
    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        tgt_len = tgt.shape[1]
        tgt_vocab_size = self.decoder.output_dim
        
        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)
        hidden, cell = self.encoder(src)
        
        input = tgt[:, 0]
        
        for t in range(1, tgt_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[:, t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = tgt[:, t] if teacher_force else top1
        
        return outputs

# ==================== 模型初始化 ====================

INPUT_DIM = src_vocab.word_count
OUTPUT_DIM = tgt_vocab.word_count
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

print("\n初始化模型...")
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
model = Seq2Seq(enc, dec, device).to(device)

print(f"模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# ==================== 训练配置 ====================

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=0)

def train_epoch(model, dataloader, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    
    progress_bar = tqdm(dataloader, desc="训练中", leave=False)
    for src, tgt in progress_bar:
        src, tgt = src.to(device), tgt.to(device)
        
        optimizer.zero_grad()
        output = model(src, tgt)
        
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        tgt = tgt[:, 1:].reshape(-1)
        
        loss = criterion(output, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        epoch_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return epoch_loss / len(dataloader)

# ==================== 训练模型 ====================

N_EPOCHS = 50
CLIP = 1

print("\n开始训练...")
print("=" * 60)

best_loss = float('inf')
for epoch in range(N_EPOCHS):
    train_loss = train_epoch(model, dataloader, optimizer, criterion, CLIP)
    
    print(f'Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f}')
    
    if train_loss < best_loss:
        best_loss = train_loss
        torch.save(model.state_dict(), 'd:/nlp/homework3/seq2seq_model.pt')

print("\n训练完成！")
print("=" * 60)

# ==================== 翻译测试 ====================

def translate_sentence(sentence, src_vocab, tgt_vocab, model, device, max_len=50):
    model.eval()
    
    tokens = src_vocab.sentence_to_indices(sentence, max_len)
    src_tensor = torch.LongTensor(tokens).unsqueeze(0).to(device)
    
    with torch.no_grad():
        hidden, cell = model.encoder(src_tensor)
    
    tgt_indices = [1]  # <sos>
    
    for _ in range(max_len):
        tgt_tensor = torch.LongTensor([tgt_indices[-1]]).to(device)
        
        with torch.no_grad():
            output, hidden, cell = model.decoder(tgt_tensor, hidden, cell)
        
        pred_token = output.argmax(1).item()
        tgt_indices.append(pred_token)
        
        if pred_token == 2:  # <eos>
            break
    
    tgt_tokens = [tgt_vocab.idx2word[i] for i in tgt_indices]
    return ' '.join(tgt_tokens[1:-1])  # 去掉<sos>和<eos>

print("\n翻译测试:")
print("=" * 60)

test_sentences = [
    "Guten Morgen",
    "Wie geht es dir",
    "Ich liebe dich",
    "Danke schön",
    "Auf Wiedersehen"
]

for src_sentence in test_sentences:
    translation = translate_sentence(src_sentence, src_vocab, tgt_vocab, model, device)
    print(f"德语: {src_sentence}")
    print(f"英语: {translation}")
    print("-" * 60)

# ==================== 保存词汇表 ====================

print("\n保存词汇表...")
with open('d:/nlp/homework3/vocabs.pkl', 'wb') as f:
    pickle.dump({'src_vocab': src_vocab, 'tgt_vocab': tgt_vocab}, f)

print("\n任务五完成！")
print("=" * 60)
print("模型已保存到: d:/nlp/homework3/seq2seq_model.pt")
print("词汇表已保存到: d:/nlp/homework3/vocabs.pkl")
