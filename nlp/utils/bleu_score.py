from collections import Counter

import numpy as np
from nltk import ngrams

""" https://wikidocs.net/31695 """


# 토큰화 된 문장(tokens)에서 n-gram을 카운트
def simple_count(tokens, n):
    return Counter(ngrams(tokens, n))


#


def count_clip(candidate: str, reference_list: str, n: int):
    # Ca 문장에서 n-gram 카운트
    ca_cnt = simple_count(candidate, n)
    # print(ca_cnt)
    max_ref_cnt_dict = dict()

    for ref in reference_list:
        # Ref 문장에서 n-gram 카운트
        ref_cnt = simple_count(ref, n)
        # 각 Ref 문장에 대해서 비교하여 n-gram의 최대 등장 횟수를 계산.
        for n_gram in ref_cnt:
            if n_gram in max_ref_cnt_dict:
                max_ref_cnt_dict[n_gram] = max(
                    ref_cnt[n_gram], max_ref_cnt_dict[n_gram]
                )
            else:
                max_ref_cnt_dict[n_gram] = ref_cnt[n_gram]

    return {
        # count_clip = min(count, max_ref_count)
        # n,gram 키가 없으면 0으로 반환
        n_gram: min(ca_cnt.get(n_gram, 0), max_ref_cnt_dict.get(n_gram, 0))
        for n_gram in ca_cnt
    }


def modified_precision(candidate, reference_list, n):
    clip_cnt = count_clip(candidate, reference_list, n)
    total_clip_cnt = sum(clip_cnt.values())  # 분자

    cnt = simple_count(candidate, n)
    total_cnt = sum(cnt.values())  # 분모

    # 분모가 0이 되는 것을 방지
    if total_cnt == 0:
        total_cnt = 1

    # 분자 : count_clip의 합, 분모 : 단순 count의 합 ==> 보정된 정밀도
    return total_clip_cnt / total_cnt


def closest_ref_length(candidate, reference_list):
    ca_len = len(candidate)  # ca 길이
    ref_lens = (len(ref) for ref in reference_list)  # Ref들의 길이
    # 길이 차이를 최소화하는 Ref를 찾아서 Ref의 길이를 리턴
    closest_ref_len = min(
        ref_lens, key=lambda ref_len: (abs(ref_len - ca_len), ref_len)
    )
    return closest_ref_len


def brevity_penalty(candidate, reference_list):
    ca_len = len(candidate)
    ref_len = closest_ref_length(candidate, reference_list)

    if ca_len > ref_len:
        return 1

    # candidate가 비어있다면 BP = 0 → BLEU = 0.0
    elif ca_len == 0:
        return 0
    else:
        return np.exp(1 - ref_len / ca_len)


def bleu_score(candidate, reference_list, weights=[0.25, 0.25, 0.25, 0.25]):
    bp = brevity_penalty(candidate, reference_list)  # 브레버티 패널티, BP

    p_n = [
        modified_precision(candidate, reference_list, n=n)
        for n, _ in enumerate(weights, start=1)
    ]
    # p1, p2, p3, ..., pn
    score = np.sum(
        [w_i * np.log(p_i) if p_i != 0 else 0 for w_i, p_i in zip(weights, p_n)]
    )
    return bp * np.exp(score)


# candidate = 'It is a guide to action which ensures that the military always obeys the commands of the party'
# references = [
#     'It is a guide to action that ensures that the military will forever heed Party commands',
#     'It is the guiding principle which guarantees the military forces always being under the command of the Party',
#     'It is the practical guide for the army always to heed the directions of the party'
# ]
candidate = "빛이 쐬는 노인은 완벽한 어두운곳에서 잠든 사람과 비교할 때 강박증이 심해질 기회가 훨씬 높았다"
references = ["빛이 쐬는 사람은 완벽한 어둠에서 잠든 사람과 비교할 때 우울증이 심해질 가능성이 훨씬 높았다"]


import nltk.translate.bleu_score as bleu

print(
    "실습 코드의 BLEU :",
    bleu_score(candidate.split(), list(map(lambda ref: ref.split(), references))),
)
print(
    "패키지 NLTK의 BLEU :",
    bleu.sentence_bleu(
        list(map(lambda ref: ref.split(), references)), candidate.split()
    ),
)

# result = bleu_score(candidate.split(), list(map(lambda ref: ref.split(), references)), n=1)
# print('보정된 유니그램 정밀도 :', result)
