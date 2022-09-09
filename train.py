from collections import Counter
import pickle
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
parser = argparse.ArgumentParser(description='Train.py')
parser.add_argument('--input-dir', type=str, help='Input dir for text', default='stdin')
parser.add_argument('--model', type=str, help='Input dir for saving model')
args = parser.parse_args()

inp = args.input_dir


class Vectorizer:
    def __init__(self, inp):
        if inp == 'stdin':
            self.text_sample = input()
        else:
            with open(inp, encoding='utf-8') as text_file:
                self.text_sample = ' '.join(text_file.readlines())
    @staticmethod
    def __preprocess_text(text):
        text = text.lower()
        text = re.sub(r"([/.,!?№#@&—;()])", r" \1 ", text)
        return text


    def text_to_seq(self):
        text_sample = self.__preprocess_text(self.text_sample)
        char_counts = Counter(text_sample)
        char_counts = sorted(char_counts.items(), key=lambda x: x[1], reverse=True)

        sorted_chars = [char for char, _ in char_counts]
        print('Словарь: ', sorted_chars)
        char_to_idx = {char: index for index, char in enumerate(sorted_chars)}
        idx_to_char = {v: k for k, v in char_to_idx.items()}
        sequence = np.array([char_to_idx[char] for char in text_sample])

        return sequence, char_to_idx, idx_to_char


vectorizer = Vectorizer(inp)

sequence, char_to_idx, idx_to_char = vectorizer.text_to_seq()

class Model(nn.Module):

    def __init__(self, seq, char_to_idx, idx_to_char, input_size, hidden_size, embedding_size, n_layers=1):
        super(Model, self).__init__()

        self.sequence = seq
        self.char_to_idx = char_to_idx
        self.idx_to_char = idx_to_char
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(self.input_size, self.embedding_size)
        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size, self.n_layers)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(self.hidden_size, self.input_size)

    def forward(self, x, hidden):
        x = self.encoder(x).squeeze(2)
        out, (ht1, ct1) = self.lstm(x, hidden)
        out = self.dropout(out)
        x = self.fc(out)
        return x, (ht1, ct1)

    def init_hidden(self, batch_size=1):
        return (torch.zeros(self.n_layers, batch_size, self.hidden_size, requires_grad=True).to(device),
                torch.zeros(self.n_layers, batch_size, self.hidden_size, requires_grad=True).to(device))

    @staticmethod
    def get_batch(sequence, batch_size, length=256):
        trains = []
        targets = []
        for _ in range(batch_size):
            batch_start = np.random.randint(0, len(sequence) - length)
            chunk = sequence[batch_start: batch_start + length]
            train = torch.LongTensor(chunk[:-1]).view(-1, 1)
            target = torch.LongTensor(chunk[1:]).view(-1, 1)
            trains.append(train)
            targets.append(target)
        return torch.stack(trains, dim=0), torch.stack(targets, dim=0)

    def fit(self, epochs, criterion, optimizer, scheduler, batch_size=16):
        loss_avg = []

        for epoch in range(epochs):
            self.train()
            train, target = self.get_batch(self.sequence, batch_size=batch_size)
            train = train.permute(1, 0, 2).to(device)
            target = target.permute(1, 0, 2).to(device)
            hidden = self.init_hidden(batch_size)

            output, hidden = self(train, hidden)
            loss = criterion(output.permute(1, 2, 0), target.squeeze(-1).permute(1, 0))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_avg.append(loss.item())
            if len(loss_avg) >= 50:
                mean_loss = np.mean(loss_avg)
                print(f'Loss: {mean_loss}')
                scheduler.step(mean_loss)
                loss_avg = []
                self.eval()

        self.eval()

    def generate(self, start_text=' ', prediction_len=200, temp=0.3):


        hidden = self.init_hidden()
        idx_input = [self.char_to_idx[char] for char in start_text]
        train = torch.LongTensor(idx_input).view(-1, 1, 1).to(device)
        predicted_text = start_text

        _, hidden = self(train, hidden)

        inp = train[-1].view(-1, 1, 1)

        for i in range(prediction_len):
            output, hidden = self(inp.to(device), hidden)
            output_logits = output.cpu().data.view(-1)
            p_next = F.softmax(output_logits / temp, dim=-1).detach().cpu().data.numpy()
            top_index = np.random.choice(len(self.char_to_idx), p=p_next)
            inp = torch.LongTensor([top_index]).view(-1, 1, 1).to(device)
            predicted_char = self.idx_to_char[top_index]
            predicted_text += predicted_char

        return predicted_text


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = Model(sequence, char_to_idx, idx_to_char, input_size=len(idx_to_char), hidden_size=128, embedding_size=128, n_layers=3)
model.to(device)

model.fit(epochs=5000,
          criterion=nn.CrossEntropyLoss(),
          optimizer=torch.optim.Adam(model.parameters(), lr=1e-2, amsgrad=True),
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        torch.optim.Adam(model.parameters(), lr=1e-2, amsgrad=True),
        patience=5,
        verbose=True,
        factor=0.5)
        )

with open(args.model, 'wb') as file:
    pickle.dump(model, file)






