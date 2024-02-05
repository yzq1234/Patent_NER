#input文件夹包括txt和ann文件，txt是文本，ann按照“序号-标签-开始位置-结束位置（左闭右开）-文本”组成
ORIGIN_DIR = './input/origin/'
ANNOTATION_DIR = './output/annotation/'

TRAIN_SAMPLE_PATH = './output/patent/patent_train.txt'
TEST_SAMPLE_PATH = './output/patent/patent_test.txt'

VOCAB_PATH = './output/patent/vocab.txt'
LABEL_PATH = './output/patent/label.txt'

WORD_PAD = '<PAD>'
WORD_UNK = '<UNK>'

VOCAB_SIZE = 5000

WORD_PAD_ID = 0
WORD_UNK_ID = 1
LABEL_O_ID = 0

EMBEDDING_DIM = 128
HIDDEN_SIZE = 128
TARGET_SIZE = 9
LR = 1e-3
EPOCH = 100

MODEL_DIR = './output/model/'

#patent
TRAIN_PATENT_SAMPLE_PATH = './output/patent/patent_train.txt'
TEST_PATENT_SAMPLE_PATH = './output/patent/patent_test.txt'

VOCAB_PATENT_PATH = './output/patent/vocab.txt'
LABEL_PATENT_PATH = './output/patent/label.txt'

#patent_car
TRAIN_PATENT_CAR_SAMPLE_PATH = './output/patent_car/train.char.bio'
TEST_PATENT_CAR_SAMPLE_PATH = './output/patent_car/test.char.bio'


VOCAB_PATENT_CAR_PATH = './output/patent_car/vocab.txt'
LABEL_PATENT_CAR_PATH = './output/patent_car/label.txt'


BERT_MODEL = './huggingface/bert-base-chinese'
EMBEDDING_DIM = 768
MAX_POSITION_EMBEDDINGS = 512