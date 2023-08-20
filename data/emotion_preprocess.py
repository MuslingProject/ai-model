from model.emotion_model import BERTDataset, tok, vocab, max_len, batch_size

# xlsx 파일을 로드한 뒤 사용할 칼럼을 정리 (감정_대분류, 사람문장1 칼럼만 사용)
# 유니크한 감정 대분류를 뽑아낸 뒤 한글 칼럼명을 숫자로 변경
import pandas as pd
import torch
from torch.utils.data import DataLoader


class Emotion:
    # 작업 디렉토리 변경
    train_set = pd.read_excel('training_data.xlsx')
    validation_set = pd.read_excel('validation_data.xlsx')

    train_set = train_set.loc[:, ['감정_대분류', '사람문장1']]
    validation_set = validation_set.loc[:, ['감정분류', '내용']]

    train_set.dropna(inplace=True)
    validation_set.dropna(inplace=True)
    train_set.columns = ['label', 'data']
    validation_set.columns = ['label', 'data']

    train_set.loc[(train_set['label'] == '사랑/기쁨'), 'label'] = 0
    train_set.loc[(train_set['label'] == '멘붕/불안'), 'label'] = 1
    train_set.loc[(train_set['label'] == '이별/슬픔'), 'label'] = 2
    train_set.loc[(train_set['label'] == '스트레스/짜증'), 'label'] = 3
    train_set.loc[(train_set['label'] == '우울'), 'label'] = 4

    validation_set.loc[(validation_set['label'] == '사랑/기쁨'), 'label'] = 0
    validation_set.loc[(validation_set['label'] == '멘붕/불안'), 'label'] = 1
    validation_set.loc[(validation_set['label'] == '이별/슬픔'), 'label'] = 2
    validation_set.loc[(validation_set['label'] == '스트레스/짜증'), 'label'] = 3
    validation_set.loc[(validation_set['label'] == '우울'), 'label'] = 4

    # 모델 학습에 사용할 데이터 셋은 [data, label] 배열로 피팅
    # 모델 학습에 사용할 훈련 데이터 셋은 기존에 주어진 train_set_data를 4:1 비율로 분리
    from sklearn.model_selection import train_test_split

    train_set_data = [[i, str(j)] for i, j in zip(train_set['data'], train_set['label'])]
    validation_set_data = [[i, str(j)] for i, j in zip(validation_set['data'], validation_set['label'])]

    train_set_data, test_set_data = train_test_split(train_set_data, test_size=0.25, random_state=0)

    train_set_data = BERTDataset(train_set_data, 0, 1, tok, vocab, max_len, True, False)
    test_set_data = BERTDataset(test_set_data, 0, 1, tok, vocab, max_len, True, False)

    # BERT 모델에 사용할 데이터 로더
    train_dataloader = torch.utils.data.DataLoader(train_set_data, batch_size=batch_size, num_workers=5)
    test_dataloader = torch.utils.data.DataLoader(test_set_data, batch_size=batch_size, num_workers=5)
