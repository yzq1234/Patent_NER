from transformers import BertModel, BertTokenizer

tokenizer = BertTokenizer.from_pretrained('./huggingface/bert-base-chinese')
model = BertModel.from_pretrained('./huggingface/bert-base-chinese')

#等长填充
batch_token1 = tokenizer(['北京欢迎你','为你开天辟地'],padding=True,return_tensors='pt' )
# print(batch_token1)
# print(batch_token1['input_ids']
encoded = model(batch_token1['input_ids'])
print(encoded[0].shape)



# #分词且编码
# token = tokenizer.encode('北京欢迎你')
# print(token)
#
# #简写形式
# token = tokenizer('北京欢迎您')
# print(token)
#
# #解码
# print(tokenizer.decode([101, 1266, 776, 3614, 6816, 2644, 102]))
#
# #查看特殊标记
# print(tokenizer.special_tokens_map)
#
# #查看特殊标记对应的id
# print(tokenizer.encode(['[UNK]','[CLS]']))



#截断
# batch_token2 = tokenizer(['北京欢迎你','为你开天辟地'],max_length=5,truncation=True)
# print(batch_token2)

# #填充刀指定长度，超过的截断
# batch_token3 = tokenizer(['北京欢迎你','为你开天辟地'],max_length=10,truncation=True,padding="max_length")
# print(batch_token3)