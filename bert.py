import torch
from transformers import BertTokenizer, BertForMaskedLM

def bert(sentence):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    sentence = sentence + " [MASK]."
    return predict_masked_sent(sentence, model, tokenizer)

def predict_masked_sent(sentence, model, tokenizer, top_k=3):
    text = "[CLS] %s [SEP]" % sentence
    tokenized_text = tokenizer.tokenize(text)
    masked_index = tokenized_text.index("[MASK]")
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])

    with torch.no_grad():
        outputs = model(tokens_tensor)
        predictions = outputs[0]

    probs = torch.nn.functional.softmax(predictions[0, masked_index], dim=-1)
    top_k_weights, top_k_indices = torch.topk(probs, top_k, sorted=True)

    prediction_list = []

    for i, pred_idx in enumerate(top_k_indices):
        predicted_token = tokenizer.convert_ids_to_tokens([pred_idx])[0]
        prediction_list.append(predicted_token)

    return prediction_list


