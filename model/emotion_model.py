import re

import gluonnlp as nlp
import torch
import numpy as np

# Hugging Face를 통한 모델 및 토크나이저 Import
from kobert_tokenizer import KoBERTTokenizer
from transformers import BertModel

# torch에서 필요 라이브러리 import
from torch.utils.data import Dataset
from model.classifier import BERTClassifier
from util.para import max_len, batch_size

# CPU 설정
ctx = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(ctx)

# BERT 모델 불러오기
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
bertmodel = BertModel.from_pretrained('skt/kobert-base-v1', return_dict=False)
vocab = nlp.vocab.BERTVocab.from_sentencepiece(tokenizer.vocab_file, padding_token='[PAD]')
tok = tokenizer.tokenize

model = BERTClassifier(bertmodel, dr_rate=0.5, num_classes=5).to(device)

model_file = "./checkpoint/SentimentAnalysisKOBert_StateDict.pt"
model.load_state_dict(torch.load(model_file, map_location=device))
model.eval()

class BERTDataset(Dataset):
  def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, vocab, max_len, pad, pair):
    transform = nlp.data.BERTSentenceTransform(
        bert_tokenizer, max_seq_length=max_len, vocab=vocab, pad=pad, pair=pair)

    self.sentences = [transform([i[sent_idx]]) for i in dataset]
    self.labels = [np.int32(i[label_idx]) for i in dataset]

  def __getitem__(self, i):
    return (self.sentences[i] + (self.labels[i], ))

  def __len__(self):
    return (len(self.labels))


# predict 함수
# 주어진 문장이 현재 학습이 완료된 모델 내에서 어떤 라벨과 argmax인지 판단하고 추론된 결과를 리턴하는 함수.
def predict(sentence):

  # 다중 문장 처리
  sentence_split = re.split(r'[,.?!~]', sentence)

  # 양쪽 공백 제거
  for sentence_i in sentence_split:
    sentence_i = sentence_i.lstrip()
    sentence_i = sentence_i.rstrip()

    # 빈 문장일 때는 종료
    if len(sentence_i) <= 0:
      result = "내용을 입력해주세요."
      return result

    # 빈 문장이 아닐 때
    else:
      dataset = [[sentence_i, '0']]
      test = BERTDataset(dataset, 0, 1, tok, vocab, max_len, True, False)
      test_dataloader = torch.utils.data.DataLoader(test, batch_size=batch_size, num_workers=0)
      model.eval()

      answer = 0

      for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)

        valid_length = valid_length
        label = label.long().to(device)

        out = model(token_ids, valid_length, segment_ids)

        test_eval = []

        for logits in out:
          logits = logits
          logits = logits.detach().cpu().numpy()

          if np.argmax(logits) == 0:
            test_eval.append("사랑/기쁨")
          elif np.argmax(logits) == 1:
            test_eval.append("멘붕/불안")
          elif np.argmax(logits) == 2:
            test_eval.append("이별/슬픔")
          elif np.argmax(logits) == 3:
            test_eval.append("스트레스/짜증")
          elif np.argmax(logits) == 4:
            test_eval.append("우울")

        result = ">> 입력하신 내용에서 " + test_eval[0] + "이/가 느껴집니다."
        return result