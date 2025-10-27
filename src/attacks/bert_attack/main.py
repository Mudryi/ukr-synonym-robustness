from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import RobertaTokenizer, RobertaConfig, RobertaForMaskedLM
from transformers import pipeline

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
import pandas as pd
import copy
import json
import string
import re
import fasttext
import numpy as np
import os

from tqdm import tqdm
from criteria import get_stopwords
import pymorphy2

morph = pymorphy2.MorphAnalyzer(lang='uk')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
filter_words = get_stopwords()

from pathlib import Path
def get_fasttext_path(env_var="FASTTEXT_UK_PATH"):
    env = os.getenv(env_var)
    if env:
        p = Path(env)
    else:
        repo_root = Path(__file__).resolve().parents[3]  # adjust depending file location
        p = repo_root / "resources" / "fasttext_uk" / "cbow.uk.300.bin"
    if not p.exists():
        raise FileNotFoundError(f"fastText model not found at {p}. Run scripts/download_fasttext_uk.sh or set {env_var}")
    return str(p)

ft_path = get_fasttext_path()
ft = fasttext.load_model(ft_path)

def compare_normal_forms(word, substitution):
    if morph.parse(word)[0].normal_form == morph.parse(substitution)[0].normal_form:
        return True
    return False


def get_data_cls(data_path, dataset, text_col='text', label_col='label'):
    df = pd.read_csv(data_path)
    if len(df)>10000:
        df = df.sample(10000, random_state=1914)

    if dataset == 'reviews':
        features = df.apply(lambda row: [row[text_col].lower(), row[label_col]-1], axis=1).tolist()
    
    elif dataset == 'news':
        label_list = ['бізнес', 'новини', 'політика', 'спорт', 'технології']
        label2id = {label: i for i, label in enumerate(label_list)}
        id2label = {i: label for label, i in label2id.items()}

        df[label_col] = df[label_col].map(label2id)

        features = df.apply(lambda row: [row[text_col].lower(), row[label_col]], axis=1).tolist()
    else:
        features = df.apply(lambda row: [row[text_col].lower(), row[label_col]], axis=1).tolist()

    return features

class Feature(object):
    def __init__(self, seq_a, label):
        self.label = label
        self.seq = seq_a
        self.final_adverse = seq_a
        self.query = 0
        self.change = 0
        self.success = 0
        self.label_after_attack = label
        self.changes = []


def evaluate(features):
    do_use = 0
    use = None
    sim_thres = 0
    # evaluate with USE

    if do_use == 1:
        cache_path = ''
        import tensorflow as tf
        import tensorflow_hub as hub
    
        class USE(object):
            def __init__(self, cache_path):
                super(USE, self).__init__()

                self.embed = hub.Module(cache_path)
                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                self.sess = tf.Session()
                self.build_graph()
                self.sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

            def build_graph(self):
                self.sts_input1 = tf.placeholder(tf.string, shape=(None))
                self.sts_input2 = tf.placeholder(tf.string, shape=(None))

                sts_encode1 = tf.nn.l2_normalize(self.embed(self.sts_input1), axis=1)
                sts_encode2 = tf.nn.l2_normalize(self.embed(self.sts_input2), axis=1)
                self.cosine_similarities = tf.reduce_sum(tf.multiply(sts_encode1, sts_encode2), axis=1)
                clip_cosine_similarities = tf.clip_by_value(self.cosine_similarities, -1.0, 1.0)
                self.sim_scores = 1.0 - tf.acos(clip_cosine_similarities)

            def semantic_sim(self, sents1, sents2):
                sents1 = [s.lower() for s in sents1]
                sents2 = [s.lower() for s in sents2]
                scores = self.sess.run(
                    [self.sim_scores],
                    feed_dict={
                        self.sts_input1: sents1,
                        self.sts_input2: sents2,
                    })
                return scores[0]

            use = USE(cache_path)

    acc = 0
    origin_success = 0
    total = 0
    total_q = 0
    total_change = 0
    total_word = 0
    for feat in features:
        if feat.success > 2:

            if do_use == 1:
                sim = float(use.semantic_sim([feat.seq], [feat.final_adverse]))
                if sim < sim_thres:
                    continue
            
            acc += 1
            total_q += feat.query
            total_change += feat.change
            total_word += len(feat.seq.split(' '))

            if feat.success == 3:
                origin_success += 1

        total += 1

    suc = float(acc / total)

    query = float(total_q / acc)
    change_rate = float(total_change / total_word)

    origin_acc = 1 - origin_success / total
    after_atk = 1 - suc

    
    with open(os.path.join('adversaries.txt'), 'w') as ofile:
        ofile.write('acc/aft-atk-acc {:.6f}/ {:.6f}, query-num {:.4f}, change-rate {:.4f}'.format(origin_acc, after_atk, query, change_rate))

    print('acc/aft-atk-acc {:.6f}/ {:.6f}, query-num {:.4f}, change-rate {:.4f}'.format(origin_acc, after_atk, query, change_rate))


def dump_features(features, output_json_path):
    outputs = []

    for feature in features:
        outputs.append({
            'label': feature.label,
            'success': feature.success,
            'change': feature.change,
            'num_word': len(feature.seq.split(' ')),
            'query': feature.query,
            'label_after_attack': feature.label_after_attack,
            'seq_a': feature.seq,
            'adv': feature.final_adverse,
        })

    # Make sure to open with UTF-8 and set ensure_ascii=False
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(outputs, f, indent=2, ensure_ascii=False)

    print('Finished dump')


def vec_or_none(word):
    try:
        return ft[word]       # uses subword model under the hood
    except KeyError:
        return None

def is_semantic_near(u, v, thr=0.35):
    vu, vv = vec_or_none(u), vec_or_none(v)
    # if either is None, fallback to keeping
    if vu is None or vv is None:
        return True
    
    nu = np.linalg.norm(vu)
    nv = np.linalg.norm(vv)
    if nu == 0 or nv == 0:
        return True   # can’t compare; let it pass
    cos = float(np.dot(vu, vv) / (nu * nv))
    return cos >= thr

def _tokenize(seq, tokenizer):
    seq = seq.replace('\n', '').lower()
    words = seq.split(' ')

    sub_words = []
    keys = []
    index = 0
    for word in words:
        sub = tokenizer.tokenize(word)
        sub_words += sub
        keys.append([index, index + len(sub)])
        index += len(sub)

    return words, sub_words, keys

def tokenize_keep_spacing(seq: str, tokenizer):
    tokens = tokenize_with_whitespace(seq.replace("\n", ""))

    sub_words = []
    keys      = []
    index     = 0
    for tok in tokens:
        # tokeniser should ignore pure whitespace -> zero subpieces
        if tok.isspace():
            keys.append([index, index])           # empty span
            continue

        pieces = tokenizer.tokenize(tok)
        sub_words.extend(pieces)
        keys.append([index, index + len(pieces)])
        index += len(pieces)

    return tokens, sub_words, keys

def _get_masked(words):
    len_text = len(words)
    masked_words = []
    for i in range(len_text - 1):
        masked_words.append(words[0:i] + ['[UNK]'] + words[i + 1:])
    # list of words
    return masked_words


def _get_masked_variants(words):
    valid_positions = [i for i, w in enumerate(words) if not filter_not_words(w)]
    masked_variants = []
    for i in valid_positions:
        tmp = list(words)
        tmp[i] = '[UNK]'
        masked_variants.append(tmp)
    return valid_positions, masked_variants


def get_important_scores(words, tgt_model, orig_prob, orig_label, orig_probs, tokenizer, batch_size, max_length):
    masked_words = _get_masked_variants(words)
    texts = [''.join(words) for words in masked_words]  # list of text of masked words
    all_input_ids = []
    all_masks = []
    for text in texts:
        inputs = tokenizer.encode_plus(text, None, add_special_tokens=True, max_length=max_length, )
        input_ids = inputs["input_ids"]
        attention_mask = [1] * len(input_ids)
        padding_length = max_length - len(input_ids)
        input_ids = input_ids + (padding_length * [0])
        attention_mask = attention_mask + (padding_length * [0])
        all_input_ids.append(input_ids)
        all_masks.append(attention_mask)
    seqs = torch.tensor(all_input_ids, dtype=torch.long)
    masks = torch.tensor(all_masks, dtype=torch.long)
    seqs = seqs.to(device)

    eval_data = TensorDataset(seqs)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)
    leave_1_probs = []
    for batch in eval_dataloader:
        masked_input, = batch
        bs = masked_input.size(0)

        leave_1_prob_batch = tgt_model(masked_input)[0]  # B num-label
        leave_1_probs.append(leave_1_prob_batch)
    leave_1_probs = torch.cat(leave_1_probs, dim=0)  # words, num-label
    leave_1_probs = torch.softmax(leave_1_probs, -1)  #
    leave_1_probs_argmax = torch.argmax(leave_1_probs, dim=-1)
    import_scores = (orig_prob
                     - leave_1_probs[:, orig_label]
                     +
                     (leave_1_probs_argmax != orig_label).float()
                     * (leave_1_probs.max(dim=-1)[0] - torch.index_select(orig_probs, 0, leave_1_probs_argmax))
                     ).data.cpu().numpy()

    return import_scores


def get_important_scores_new(words, tgt_model, orig_prob, orig_label, orig_probs, tokenizer, batch_size, max_length):
    valid_positions, masked_wordlists = _get_masked_variants(words)

    all_input_ids = []

    for wlist in masked_wordlists:
        text = ''.join(wlist)  # your words include spaces, so no extra delimiter
        enc = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
            padding='max_length'
        )
        all_input_ids.append(enc["input_ids"])
    
    seqs = torch.tensor(all_input_ids, dtype=torch.long).to(device)

    eval_data = TensorDataset(seqs)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)

    leave_1_probs = []
    for batch in eval_dataloader:
        masked_input, = batch
        leave_1_prob_batch = tgt_model(masked_input)[0]  # B num-label
        leave_1_probs.append(leave_1_prob_batch)

    if not leave_1_probs:
        # defensive: in case DataLoader somehow was empty
        return np.zeros(len(words), dtype=float)

    leave_1_probs = torch.cat(leave_1_probs, dim=0)  # words, num-label
    leave_1_probs = torch.softmax(leave_1_probs, -1)  #
    leave_1_probs_argmax = torch.argmax(leave_1_probs, dim=-1)
    raw_scores = (orig_prob
                     - leave_1_probs[:, orig_label]
                     +
                     (leave_1_probs_argmax != orig_label).float()
                     * (leave_1_probs.max(dim=-1)[0] - torch.index_select(orig_probs, 0, leave_1_probs_argmax))
                     ).data.cpu().numpy()
    
    import_scores = np.zeros(len(words), dtype=float)
    for pos, sc in zip(valid_positions, raw_scores):
        import_scores[pos] = sc

    return import_scores


def get_substitues(substitutes, tokenizer, mlm_model, use_bpe, substitutes_score=None, threshold=3.0):
    words = []
    sub_len, k = substitutes.size()

    if sub_len == 0:
        return words
        
    elif sub_len == 1:
        for (tok_id,j) in zip(substitutes[0], substitutes_score[0]):
            if threshold != 0 and j < threshold:
                break
            token = tokenizer._convert_id_to_token(int(tok_id))
            
            if not token.startswith("▁"):            
                continue
            clean_word = token.lstrip("▁")            # drop the marker
            words.append(clean_word)
    elif sub_len > 4:
        return words
    else:
        if use_bpe == 1:
            words = get_bpe_substitues(substitutes, tokenizer, mlm_model)
        else:
            return words
    
    return words


def get_bpe_substitues(substitutes, tokenizer, mlm_model, k_sub=12):
    substitutes = substitutes[:, :k_sub]

    all_substitutes = []
    for i in range(substitutes.size(0)):
        if len(all_substitutes) == 0:
            lev_i = substitutes[i]
            all_substitutes = [[int(c)] for c in lev_i]
        else:
            lev_i = []
            for all_sub in all_substitutes:
                for j in substitutes[i]:
                    lev_i.append(all_sub + [int(j)])
            all_substitutes = lev_i

    c_loss = nn.CrossEntropyLoss(reduction='none')
    word_list = []
    all_substitutes = torch.tensor(all_substitutes) 
    all_substitutes = all_substitutes[:64].to(device)
    N, L = all_substitutes.size()
    word_predictions = mlm_model(all_substitutes)[0] 
    ppl = c_loss(word_predictions.view(N*L, -1), all_substitutes.view(-1)) 
    ppl = torch.exp(torch.mean(ppl.view(N, L), dim=-1))
    _, word_list = torch.sort(ppl)
    word_list = [all_substitutes[i] for i in word_list]
    final_words = []
    for word in word_list:
        tokens = [tokenizer._convert_id_to_token(int(i)) for i in word]
        try:
            text = tokenizer.convert_tokens_to_string(tokens)
            final_words.append(text)
        except:
            print("Invalid tokens:", tokens)
            final_words.append("[UNK]")

    final_words   = list(dict.fromkeys(final_words))
    return final_words


_lat_pat = re.compile(r'[A-Za-z]')
_rus_exclusive_pat = re.compile(r'[ЁёЫыЭэЪъ]')
def has_foreign_letters(token: str) -> bool:
    return bool(_lat_pat.search(token) or _rus_exclusive_pat.search(token))


def filter_not_words(word, target_word=None):
    if word.lower() in filter_words:
        return True

    if not any(ch.isalpha() for ch in word):
        return True
    
    if len(word) < 3:
        return True
    
    if ("</s>" in word) or ("<s>" in word):
        return True
    
    if target_word is not None and word == target_word:
        return True

    return False


def tokenize_with_whitespace(text):
    pattern = r'(\s+|[^\w\s]+|\w+)'
    tokens = re.findall(pattern, text)
    return tokens


def attack(feature, tgt_model, mlm_model, tokenizer_tgt, tokenizer_mlm, k, batch_size, max_length=512, use_bpe=1, threshold_pred_score=0.3,
           cos_sim_threshold=0.4):
    words = tokenize_with_whitespace(feature.seq)

    inputs = tokenizer_tgt.encode_plus(feature.seq, None, add_special_tokens=True, max_length=max_length)
    input_ids = torch.tensor(inputs["input_ids"])

    attention_mask = torch.tensor([1] * len(input_ids))
    orig_probs = tgt_model(input_ids.unsqueeze(0).to(device),
                           attention_mask.unsqueeze(0).to(device))[0].squeeze()
    orig_probs = torch.softmax(orig_probs, -1)
    orig_label = torch.argmax(orig_probs)
    current_prob = orig_probs.max()

    if orig_label != feature.label:
        feature.success = 3
        return feature
    
    important_scores = get_important_scores_new(words, tgt_model, current_prob, orig_label, orig_probs, tokenizer_tgt, batch_size, max_length)
    feature.query += int(len(words)/2)

    list_of_index = sorted(enumerate(important_scores), key=lambda x: x[1], reverse=True)
    final_words = copy.deepcopy(words)

        
    _, sub_words_mlm, keys_mlm = tokenize_keep_spacing(feature.seq, tokenizer_mlm)

    sub_words_mlm = ['<s>'] + sub_words_mlm[:max_length - 2] + ['</s>']
    input_ids_ = torch.tensor([tokenizer_mlm.convert_tokens_to_ids(sub_words_mlm)])
    word_predictions = mlm_model(input_ids_.to(device))[0].squeeze()  # seq-len(sub) vocab
    word_pred_scores_all, word_predictions = torch.topk(word_predictions, k, -1)  # seq-len k

    word_predictions = word_predictions[1:len(sub_words_mlm) + 1, :]
    word_pred_scores_all = word_pred_scores_all[1:len(sub_words_mlm) + 1, :]

    for top_index in list_of_index:
        if feature.change > int(0.3 * (len(words)/2)):
            feature.success = 1  # exceed
            return feature

        tgt_word = words[top_index[0]]

        if filter_not_words(tgt_word):
            continue

        if keys_mlm[top_index[0]][0] > max_length - 2:
            continue

        substitutes = word_predictions[keys_mlm[top_index[0]][0]:keys_mlm[top_index[0]][1]]  # L, k
        word_pred_scores = word_pred_scores_all[keys_mlm[top_index[0]][0]:keys_mlm[top_index[0]][1]]

        substitutes = get_substitues(substitutes, tokenizer_mlm, mlm_model, use_bpe, word_pred_scores, threshold_pred_score)

        most_gap = 0.0
        candidate = None
        
        for substitute_ in substitutes:
            if substitute_ is None:
                print('None substitute')
                continue

            substitute = substitute_

            if has_foreign_letters(substitute):
                continue
            
            if filter_not_words(substitute_, tgt_word):
                continue

            if not is_semantic_near(tgt_word, substitute, cos_sim_threshold):
                continue

            if compare_normal_forms(tgt_word, substitute):
                continue
                
            temp_replace = final_words
            temp_replace[top_index[0]] = substitute
            temp_text = ''.join(temp_replace)

            inputs = tokenizer_tgt.encode_plus(temp_text, None, add_special_tokens=True, max_length=max_length)
            input_ids = torch.tensor(inputs["input_ids"]).unsqueeze(0).to(device)
            temp_prob = tgt_model(input_ids)[0].squeeze()
            feature.query += 1
            temp_prob = torch.softmax(temp_prob, -1)
            temp_label = torch.argmax(temp_prob)

            if temp_label != orig_label:
                feature.change += 1
                final_words[top_index[0]] = substitute
                feature.changes.append([keys_mlm[top_index[0]][0], substitute, tgt_word])
                feature.final_adverse = temp_text
                feature.success = 4
                return feature
            else:

                label_prob = temp_prob[orig_label]
                gap = current_prob - label_prob
                if gap > most_gap:
                    most_gap = gap
                    candidate = substitute

        if most_gap > 0:
            feature.change += 1
            feature.changes.append([keys_mlm[top_index[0]][0], candidate, tgt_word])
            current_prob = current_prob - most_gap
            final_words[top_index[0]] = candidate

    feature.final_adverse = (''.join(final_words))
    feature.success = 2
    return feature


def attack_via_mask(feature, tgt_model, tokenizer_tgt, unmasker, tokenizer_mlm, max_length=512, num_subs=128, threshold_score=0.04, cos_sim_threshold=0.33, batch_size=64):
    # get original probability
    inputs = tokenizer_tgt.encode_plus(feature.seq, None, add_special_tokens=True, max_length=max_length)
    input_ids = torch.tensor(inputs["input_ids"])
    attention_mask = torch.tensor(inputs["attention_mask"]) 
    orig_probs = tgt_model(input_ids.unsqueeze(0).to(device),
                        attention_mask.unsqueeze(0).to(device))[0].squeeze()
    orig_probs = torch.softmax(orig_probs, -1)
    true_label = torch.argmax(orig_probs)
    true_prob = orig_probs.max()

    if true_label != feature.label:
        feature.success = 3
        return feature

    tokens = tokenizer_mlm.encode_plus(''.join(feature.seq),
                                           return_tensors=None,
                                           return_attention_mask=False,
                                           return_token_type_ids=False)
    if len(tokens['input_ids'])>511:
        print("too long feature")
        return feature
    
    words = tokenize_with_whitespace(feature.seq)
    final_words = copy.deepcopy(words)

    important_scores = get_important_scores_new(words, tgt_model, true_prob, true_label, orig_probs, tokenizer_tgt, batch_size, max_length)
    feature.query += int(len(words)/2)
    
    sorted_positions = np.argsort(important_scores)[::-1]

    # for i in range(len(words)): # TODO investigate how BERt attack get importanc scores
    for i in sorted_positions:

        if feature.change > int(0.4 * (len(words)/2)):
            feature.success = 1  # exceed
            return feature

        if filter_not_words(words[i]):
            continue

        masked_text = ["<mask>" if idx == i else word for idx, word in enumerate(words)]

        mask_results = unmasker(''.join(masked_text), top_k=num_subs)

        filtered = [r for r in mask_results if r["score"] >= threshold_score]

        unmasked_words = [j['token_str'] for j in filtered]

        most_gap = 0.0
        candidate = None

        for new_word in unmasked_words:

            if has_foreign_letters(new_word):
                continue
    
            if filter_not_words(new_word, words[i]):
                continue

            if not is_semantic_near(words[i], new_word, cos_sim_threshold):
                continue

            if compare_normal_forms(words[i], new_word):
                continue

            replaced_text = "".join([new_word if idx == i else word for idx, word in enumerate(final_words)])

            inputs = tokenizer_tgt.encode_plus(replaced_text, None, add_special_tokens=True, max_length=max_length)
            input_ids = torch.tensor(inputs["input_ids"])
            attention_mask = torch.tensor(inputs["attention_mask"]) #torch.tensor([1] * len(input_ids))
            adv_probs = tgt_model(input_ids.unsqueeze(0).to(device),
                                 attention_mask.unsqueeze(0).to(device))[0].squeeze()
            feature.query += 1
            adv_probs = torch.softmax(adv_probs, -1)
            adv_label = torch.argmax(adv_probs)
            current_prob = adv_probs.max()

            if adv_label != true_label:
                feature.change += 1
                final_words[i] = new_word
                feature.changes.append([new_word, words[i]])
                feature.final_adverse = replaced_text
                feature.success = 4
                feature.label_after_attack = adv_label.item()
                return feature
            else:
                label_prob = adv_probs[true_label]
                gap = true_prob - label_prob
                if gap > most_gap:
                    most_gap = gap
                    candidate = new_word

        if most_gap > 0:
            feature.change += 1
            feature.changes.append([candidate, words[i]])
            true_prob = true_prob - most_gap
            final_words[i] = candidate

    feature.final_adverse = (''.join(final_words))
    feature.success = 2
    return feature


def run_attack():
    mlm_path = "FacebookAI/xlm-roberta-large" #'xlm-roberta-base'
    use_bpe = 1
    k = 48
    threshold_pred_score = 60

    tokenizer_mlm = AutoTokenizer.from_pretrained(mlm_path, truncation=True)
    tokenizer_mlm.model_max_length = 512
    config_atk = RobertaConfig.from_pretrained(mlm_path)
    mlm_model = RobertaForMaskedLM.from_pretrained(mlm_path, config=config_atk)
    mlm_model.to(device)

    unmasker = pipeline('fill-mask', model=mlm_path, tokenizer=tokenizer_mlm)


    data_path = '/home/mudryi/phd_projects/xml-roberta-finetune-reviews/ua-news/test.csv'
    num_label = 5

    # tgt_path = "/home/mudryi/phd_projects/xml-roberta-finetune-reviews/trained_models/tmdk/model_tmdk_7_600"
    # tgt_tokenizer = "xlm-roberta-base" #"youscan/ukr-roberta-base" "xlm-roberta-base" "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

    for tgt_path, tgt_tokenizer in zip(['/home/mudryi/phd_projects/xml-roberta-finetune-reviews/trained_models/npz4/model_npz4_9_1000',
                                        '/home/mudryi/phd_projects/xml-roberta-finetune-reviews/trained_models/1kjq/model_1kjq_9_2500',
                                        '/home/mudryi/phd_projects/xml-roberta-finetune-reviews/trained_models/3rzr/model_3rzr_9_2500'
                                        ],
                                       ["youscan/ukr-roberta-base", 
                                        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2", 
                                        "xlm-roberta-base"]):
        print(f"Start processing {tgt_tokenizer}")
        features = get_data_cls(data_path, dataset='news', text_col='title', label_col='target') #text_col='title', label_col='target'

        output_dir = "adv_results_news_"+tgt_tokenizer.split('/')[0]

        tokenizer_tgt = AutoTokenizer.from_pretrained(tgt_tokenizer)
        tgt_model = AutoModelForSequenceClassification.from_pretrained(tgt_path, num_labels=num_label)
        tgt_model.to(device)

        features_output = []
        with torch.no_grad():
            for index, feature in tqdm(enumerate(features), total=len(features)):
                seq_a, label = feature
                feat = Feature(seq_a, label)
                # feat = attack(feat, tgt_model, mlm_model, tokenizer_tgt, tokenizer_mlm, k, batch_size=64, max_length=512,
                #               use_bpe=use_bpe,threshold_pred_score=threshold_pred_score)
                feat = attack_via_mask(feat, tgt_model, tokenizer_tgt, unmasker, tokenizer_mlm)

                features_output.append(feat)

        evaluate(features_output)

        dump_features(features_output, output_dir)


if __name__ == '__main__':
    run_attack() 