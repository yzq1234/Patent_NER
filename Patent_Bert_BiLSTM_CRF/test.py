from utils import *
from Bert_model import *
from config import *

if __name__ == '__main__':
    # 检查是否有可用的GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = Dataset('test')
    loader = data.DataLoader(dataset, batch_size=64, collate_fn=collate_fn)

    id2label,_ = get_label()
    with torch.no_grad():
        y_true_list = []
        y_pred_list = []
        model = torch.load(MODEL_DIR + 'model_90.path').to(device)

        for b, (input, target, mask) in enumerate(loader):
            input, target, mask = input.to(device), target.to(device), mask.to(device)
            y_pred = model(input, mask)
            loss = model.loss_fn(input, target, mask)
            # 拼接返回值
            for lst in y_pred:
                y_pred_list.append([id2label[i] for i in lst])
            for y, m in zip(target, mask):
                y_true_list.append([id2label[i] for i in y[m==True].tolist()])

            print('>> batch:', b, 'loss:', loss.item())

        # 整体准确率
        # y_true_tensor = torch.tensor(y_true_list)
        # y_pred_tensor = torch.tensor(y_pred_list)
        # accuracy = (y_true_tensor == y_pred_tensor).sum() / len(y_true_tensor)
        # print('>> total:', len(y_true_tensor), 'accuracy:', accuracy.item())

        print(report(y_true_list,y_pred_list))