import torch
from torch.utils import data
from config import *
import pandas as pd
from seqeval.metrics import classification_report

def report(y_true, y_pred):
    return classification_report(y_true, y_pred)



from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

def detailed_report(y_true, y_pred):
    # Flatten the true and predicted labels
    y_true_flat = [label for sublist in y_true for label in sublist]
    y_pred_flat = [label for sublist in y_pred for label in sublist]

    # Get a list of unique entity types (excluding the 'O' label)
    entity_types = set([label[2:] for label in y_true_flat if label != 'O'])

    # Initialize dictionaries to store results for each entity type
    precision_dict = {}
    recall_dict = {}
    f1_dict = {}

    # Calculate precision, recall, and F1 for each entity type
    for entity_type in entity_types:
        y_true_entity = [[label] for label in y_true_flat if label.endswith(entity_type)]
        y_pred_entity = [[label] for label in y_pred_flat if label.endswith(entity_type)]

        precision = precision_score(y_true_entity, y_pred_entity)
        recall = recall_score(y_true_entity, y_pred_entity)
        f1 = f1_score(y_true_entity, y_pred_entity)

        precision_dict[entity_type] = precision
        recall_dict[entity_type] = recall
        f1_dict[entity_type] = f1

        print(f"Entity Type: {entity_type}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("---------------")

    # Calculate overall precision, recall, and F1
    overall_precision = precision_score([y_true_flat], [y_pred_flat])
    overall_recall = recall_score([y_true_flat], [y_pred_flat])
    overall_f1 = f1_score([y_true_flat], [y_pred_flat])

    print("Overall Metrics:")
    print(f"Overall Precision: {overall_precision:.4f}")
    print(f"Overall Recall: {overall_recall:.4f}")
    print(f"Overall F1 Score: {overall_f1:.4f}")

    # Return a classification report for the overall evaluation
    return classification_report([y_true_flat], [y_pred_flat])




def get_vocab():
    df = pd.read_csv(VOCAB_PATH,names=['word','id'])
    return list(df["word"]),dict(df.values)

def get_label():
    df = pd.read_csv(LABEL_PATH,names=['label','id'])
    return list(df['label']),dict(df.values)


class Dataset(data.Dataset):
    def __init__(self,type='train',base_len=20):
        super().__init__()
        self.base_len = base_len
        sample_path = TRAIN_SAMPLE_PATH if type == 'train' else TEST_SAMPLE_PATH
        #df是txt
        self.df = pd.read_csv(sample_path,names=['word','label'])
        _,self.word2id = get_vocab()
        _,self.label2id = get_label()
        self.get_points()

    #获取数据集的分割点列表
    def get_points(self):
        self.points = [0]
        i = 0
        while True:
            if i + self.base_len >= len(self.df):
                self.points.append(len(self.df))
                break
            if self.df.loc[i+self.base_len,'label'] == 'O':
                i += self.base_len
                self.points.append(i)
            else:
                i+=1

    def __len__(self):
        return len(self.points) - 1


    def __getitem__(self, index):
        #每句话 （word 和 label）
        df = self.df[self.points[index]:self.points[index+1]]
        word_unk_id = self.word2id[WORD_UNK]
        label_o_id = self.label2id['O']
        input = [self.word2id.get(w,word_unk_id) for w in df['word']]
        target = [self.label2id.get(l,label_o_id) for l in df['label']]
        return input,target

#填充函数
def collate_fn(batch):
    batch.sort(key=lambda x:len(x[0]),reverse = True)
    max_len = len(batch[0][0])
    input = []
    target = []
    mask = []
    for item in batch:
        pad_len = max_len - len(item[0])
        input.append(item[0] + [WORD_PAD_ID] * pad_len)
        target.append(item[1] + [LABEL_O_ID] *pad_len)
        mask.append([1] * len(item[0]) + [0] *pad_len)
    return  torch.tensor(input),torch.tensor(target),torch.tensor(mask).bool()

if __name__ == '__main__':
    dataset = Dataset()
    #每个loader有10句话，10句标签，10个mask,每个都是二维
    loader = data.DataLoader(dataset,batch_size = 2,collate_fn=collate_fn)
    print(iter(loader).next())