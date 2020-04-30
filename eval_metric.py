# ------  Defining evaluation metric  ------
def intersection(lst1, lst2):
    """
    Get intersection of two lists.
    :param lst1: first list
    :param lst2: second list
    :return: list containing intersection
    """
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def match_M(batch_scores_no_padd, batch_labels_no_pad):
    """
    Compute score.
    :param batch_scores_no_padd: predicted scores of the batch without padding
    :param batch_labels_no_padd: actual labels of the batch without padding
    :return: batch_num_m: number of words considered for m=[1,2,3,4] while evaluating score
             batch_score_m: total score of words considered for m=[1,2,3,4]
    """
    top_m = [1, 2, 3, 4]
    batch_num_m=[]
    batch_score_m=[]
    for m in top_m:
        intersects_lst = []
        # exact_lst = []
        score_lst = []
        ### computing scores:
        for s in batch_scores_no_padd:
            if len(s) <=m:
                continue
            h = m
            s = np.asarray(s)
            ind_score = sorted(range(len(s)), key = lambda sub: s[sub])[-h:]
            score_lst.append(ind_score)

        ### computing labels:
        label_lst = []
        for l in batch_labels_no_pad:
            if len(l) <=m:
                continue
            h = m
            if len(l) > h:
                while (l[np.argsort(l)[-h]] == l[np.argsort(l)[-(h + 1)]] and h < (len(l) - 1)):
                    h += 1
            l = np.asarray(l)
            ind_label = np.argsort(l)[-h:]
            label_lst.append(ind_label)

        ### :

        for i in range(len(score_lst)):
            intersect = intersection(score_lst[i], label_lst[i])
            intersects_lst.append((len(intersect))/(min(m, len(score_lst[i]))))
        batch_num_m.append(len(score_lst))
        batch_score_m.append(sum(intersects_lst))
    return batch_num_m, batch_score_m
# ------------------------------------------

# Fix the padding of predicted labels
def fix_padding(scores_numpy, label_probs, mask_numpy):
    """
    Fixes the padding
    :param scores_numpy: predicted scores
    :param label_probs: actual probs
    :param mask_numpy: mask
    :return: scores and labels with no padding
    """
    all_scores_no_padd = []
    all_labels_no_pad = []
    for i in range(len(mask_numpy)):
        all_scores_no_padd.append(scores_numpy[i][:int(mask_numpy[i])])
        all_labels_no_pad.append(label_probs[i][:int(mask_numpy[i])])

    assert len(all_scores_no_padd) == len(all_labels_no_pad)
    return all_scores_no_padd, all_labels_no_pad