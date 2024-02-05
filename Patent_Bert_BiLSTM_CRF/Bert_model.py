import torch.nn as nn
from config import *
from torchcrf import CRF
import torch
from transformers import BertModel

class Model(nn.Module):
    def __init__(self, dropout_rate=0.4):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM, WORD_PAD_ID)
        self.bert = BertModel.from_pretrained(BERT_MODEL)
        #self.embedding_dropout = nn.Dropout(dropout_rate)
        self.lstm = nn.LSTM(
            EMBEDDING_DIM,
            HIDDEN_SIZE,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate  # Add dropout to the LSTM layer
        )
        self.lstm_dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(2 * HIDDEN_SIZE, TARGET_SIZE)
        self.crf = CRF(TARGET_SIZE, batch_first=True)

    def _get_lstm_feature(self, input,mask):
        # out = self.embed(input)
        # out = self.embedding_dropout(out)
        out = self.bert(input, mask)[0]
        out, _ = self.lstm(out)
        out = self.lstm_dropout(out)
        return self.linear(out)

    def forward(self, input, mask):
        out = self._get_lstm_feature(input,mask)
        return self.crf.decode(out, mask)

    def loss_fn(self, input, target, mask):
        y_pred = self._get_lstm_feature(input,mask)
        return -self.crf.forward(y_pred, target, mask, reduction='mean')

if __name__ == '__main__':
    model = Model()
    # 100个句子，50个词
    input = torch.randint(1, 3000, (100, 50))
    print(model(input).shape)
