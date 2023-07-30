import random
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from time import time
from prettytable import PrettyTable

import os
import logging
from utils.parser import parse_args
from utils.data_loader import load_data
from modules.DCLKR import Recommender
from utils.helper import early_stopping, ensureDir

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

n_users = 0
n_items = 0
n_entities = 0
n_nodes = 0
n_relations = 0

def get_contra_feed_dict(index, start, end):
    index = index[start:end]
    users = torch.LongTensor(index[index < n_users]).to(device)
    entities = torch.LongTensor(index[index >= n_users] - n_users).to(device)
    feed_dict = {}
    feed_dict['users'] = users
    feed_dict['entities'] = entities
    feed_dict['phase'] = 1
    return feed_dict

def get_feed_dict(train_entity_pairs, start, end):
    train_entity_pairs = torch.LongTensor(np.array([[cf[0], cf[1], cf[2]] for cf in train_entity_pairs], np.int32))
    feed_dict = {}
    entity_pairs = train_entity_pairs[start:end].to(device)
    feed_dict['users'] = entity_pairs[:, 0]
    feed_dict['items'] = entity_pairs[:, 1]
    feed_dict['labels'] = entity_pairs[:, 2]
    return feed_dict

def get_feed_dict_topk(train_entity_pairs, start, end):
    train_entity_pairs = torch.LongTensor(np.array([[cf[0], cf[1], cf[2]] for cf in train_entity_pairs], np.int32))
    feed_dict = {}
    entity_pairs = train_entity_pairs[start:end].to(device)
    feed_dict['users'] = entity_pairs[:, 0]
    feed_dict['items'] = entity_pairs[:, 1]
    feed_dict['labels'] = entity_pairs[:, 2]
    return feed_dict

def _show_recall_info(recall_zip):
    res = ""
    for i, j in recall_zip:
        res += "K@%d:%.4f  "%(i,j)
    logging.info(res)

def _get_topk_feed_data(user, items):
    res = list()
    for item in items:
        res.append([user, item, 0])
    return np.array(res)

def _get_user_record(data, is_train):
    user_history_dict = dict()
    for rating in data:
        user = rating[0]
        item = rating[1]
        label = rating[2]
        if is_train or label == 1:
            if user not in user_history_dict:
                user_history_dict[user] = set()
            user_history_dict[user].add(item)
    return user_history_dict

def ctr_eval(model, data):
    auc_list = []
    f1_list = []
    start = 0

    model.eval()
    with torch.no_grad():
        entity_embs, user_embs = model.generate()
    
    while start < data.shape[0]:
        batch = get_feed_dict(data, start, start + args.batch_size)
        labels = data[start:start + args.batch_size, 2]
        with torch.no_grad():
            scores = model.generate_scores(batch, entity_embs, user_embs)
        scores = scores.detach().cpu().numpy()
        auc = roc_auc_score(y_true=labels, y_score=scores)
        predictions = [1 if i >= 0.5 else 0 for i in scores]
        f1 = f1_score(y_true=labels, y_pred=predictions)
        auc_list.append(auc)
        f1_list.append(f1)
        start += args.batch_size
    
    model.train()
    auc = float(np.mean(auc_list))
    f1 = float(np.mean(f1_list))
    return auc, f1

def topk_eval(model, train_data, data):
    # logging.info('calculating recall ...')
    k_list = [5, 10, 20, 50, 100]
    recall_list = {k: [] for k in k_list}
    item_set = set(train_data[:, 1].tolist() + data[:, 1].tolist())
    train_record = _get_user_record(train_data, True)
    test_record = _get_user_record(data, False)
    user_list = list(set(train_record.keys()) & set(test_record.keys()))
    user_num = 100
    if len(user_list) > user_num:
        np.random.seed()
        user_list = np.random.choice(user_list, size=user_num, replace=False)

    model.eval()
    with torch.no_grad():
        entity_embs, user_embs = model.generate()
    
    for user in user_list:
        test_item_list = list(item_set-set(train_record[user]))
        item_score_map = dict()
        start = 0
        while start + args.batch_size <= len(test_item_list):
            items = test_item_list[start:start + args.batch_size]
            input_data = _get_topk_feed_data(user, items)
            batch = get_feed_dict_topk(input_data, start, start + args.batch_size)
            with torch.no_grad():
                scores = model.generate_scores(batch, entity_embs, user_embs)
            for item, score in zip(items, scores):
                item_score_map[item] = score
            start += args.batch_size
        # padding the last incomplete mini-batch if exists
        if start < len(test_item_list):
            res_items = test_item_list[start:] + [test_item_list[-1]] * (args.batch_size - len(test_item_list) + start)
            input_data = _get_topk_feed_data(user, res_items)
            batch = get_feed_dict_topk(input_data, start, start + args.batch_size)
            with torch.no_grad():
                scores = model.generate_scores(batch, entity_embs, user_embs)
            for item, score in zip(res_items, scores):
                item_score_map[item] = score
        item_score_pair_sorted = sorted(item_score_map.items(), key=lambda x: x[1], reverse=True)
        item_sorted = [i[0] for i in item_score_pair_sorted]
        for k in k_list:
            hit_num = len(set(item_sorted[:k]) & set(test_record[user]))
            recall_list[k].append(hit_num / len(set(test_record[user])))
    model.train()

    recall = [np.mean(recall_list[k]) for k in k_list]
    return recall
    # _show_recall_info(zip(k_list, recall))

if __name__ == '__main__':
    """fix the random seed"""
    seed = 2020
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    """read args"""
    global args, device
    args = parse_args()
    device = torch.device("cuda:"+str(args.gpu_id)) if args.cuda else torch.device("cpu")

    """build dataset"""
    train_cf, test_cf, user_dict, n_params, graph, mat_list, interaction = load_data(args)
    adj_mat_list, norm_mat_list, mean_mat_list = mat_list

    n_users = n_params['n_users']
    n_items = n_params['n_items']
    n_entities = n_params['n_entities']
    n_relations = n_params['n_relations']
    n_nodes = n_params['n_nodes']

    """cf data"""
    # train_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1], cf[2]] for cf in train_cf], np.int32))
    # eval_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1], cf[2]] for cf in eval_cf], np.int32))
    # LongTensor
    # labels = torch.FloatTensor(np.array(cf[2] for cf in train_cf))
    test_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1], cf[2]] for cf in test_cf], np.int32))

    """define model"""
    model = Recommender(n_params, args, graph, mean_mat_list[0]).to(device)

    """define optimizer"""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    cur_best_pre_0 = 0
    stopping_step = 0
    should_stop = False

    print("start training ...")
    for epoch in range(args.epoch):
        train_s_t = time()

        """training CF"""
        # shuffle training data
        index = np.arange(len(train_cf))
        np.random.shuffle(index)
        train_cf = train_cf[index]

        """training"""
        s, loss, contra_loss, contra_loss_2 = 0, 0, 0, 0
        while s + args.batch_size <= len(train_cf):
            batch = get_feed_dict(train_cf, s, s + args.batch_size)
            batch_loss, _, _, _, batch_contra, batch_contra2 = model(batch)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss
            contra_loss += batch_contra
            contra_loss_2 += batch_contra2
            s += args.batch_size

        model.update_interact_mats()

        train_e_t = time()
        if epoch % 5 == 4 or epoch == 1:
            """testing"""
            test_s_t = time()
            test_auc, test_f1 = ctr_eval(model, test_cf_pairs)
            test_e_t = time()
            train_res = PrettyTable()
            train_res.field_names = ["Epoch", "training time", "tesing time", "Loss", "Contra", "test auc", "test f1"]
            train_res.add_row(
                    [epoch, train_e_t - train_s_t, test_e_t - test_s_t, loss.item(), contra_loss.item(), test_auc, test_f1]
                )
            print(train_res)

            test_s_t = time()
            recall = topk_eval(model, train_cf, test_cf)
            test_e_t = time()
            train_res = PrettyTable()
            train_res.field_names = ["Recall@5", "Recall@10", "Recall@20", "Recall@50", "Recall@100"]
            train_res.add_row(recall)
            print(train_res)

            # *********************************************************
            # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
            cur_best_pre_0, stopping_step, should_stop = early_stopping(test_auc, cur_best_pre_0,
                                                                        stopping_step, expected_order='acc',
                                                                        flag_step=10)
            if should_stop:
                break

            """save weight"""
            if test_auc == cur_best_pre_0 and args.save:
                save_path = args.out_dir + 'DCLKR_' + args.dataset + '.ckpt'
                ensureDir(save_path)
                torch.save(model.state_dict(), save_path)
                final_entity_emb, final_user_emb = model.generate()
        else:
            print('using time %.4f, training loss at epoch %d: %.4f, contra: %.4f, contra2: %.4f' % (train_e_t - train_s_t, epoch, loss.item(), contra_loss.item(), contra_loss_2.item()))

    print('early stopping at %d, test_auc:%.4f' % (epoch-30, cur_best_pre_0))

