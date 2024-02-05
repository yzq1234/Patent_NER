import torch

from utils import *
from Bert_model import *
from config import *

if __name__ == '__main__':
    # 检查是否有可用的GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = Dataset()
    loader = data.DataLoader(
        dataset,
        batch_size = 1,
        shuffle = True,
        collate_fn = collate_fn
    )

    model = Model().to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=LR)

    for e in range(EPOCH):
        for b,(input,target,mask) in enumerate(loader):
            input, target, mask = input.to(device), target.to(device), mask.to(device)
            y_pred = model(input,mask)
            loss = model.loss_fn(input,target,mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if b % 10 ==0:
                print('>>epoch:',e,'loss:',loss.item())

        torch.save(model,MODEL_DIR+f'model_{e}.path')