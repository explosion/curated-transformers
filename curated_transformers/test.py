from transformers import AutoModel

bert = AutoModel.from_pretrained("explosion-testing/bert-test")
electra = AutoModel.from_pretrained("jonfd/electra-small-nordic")


electra