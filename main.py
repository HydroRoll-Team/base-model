from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained("hfl/minirbt-h256")
model = BertModel.from_pretrained("hfl/minirbt-h256")