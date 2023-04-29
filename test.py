import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import transformers 
from transformers import AutoTokenizer,BertModel
# s = "hello world I am using bert"
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# enc = tokenizer(s)
# ids = torch.tensor([enc["input_ids"]])
# mask = torch.tensor([enc["attention_mask"]])
# token_type_ids = torch.tensor([enc["token_type_ids"]])
# model = BertModel.from_pretrained('bert-base-uncased')
# print(type(model.encoder.layer[:2]))
a = torch.tensor([1,1,4,5,1,4,0,0,0])
print((a!=0).long())
