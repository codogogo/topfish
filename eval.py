import codecs
from scipy import stats
import os
import numpy as np
import sys

def pairwise_accuracy(golds, preds):
	count_good = 0.0
	count_all = 0.0
	for i in range(len(golds) - 1):
		for j in range(i+1, len(golds)):
			count_all += 1.0
			diff_gold = golds[i] - golds[j]
			diff_pred = preds[i] - preds[j]
			if (diff_gold * diff_pred >= 0):
				count_good += 1.0
	return count_good / count_all


def evaluate(gold_path, predicted_path):
	golds = [(x.split()[0].strip(), float(x.split()[1].strip())) for x in list(codecs.open(gold_path, "r", "utf-8").readlines())]
	predicts =  [(x.split()[0].strip(), float(x.split()[1].strip())) for x in list(codecs.open(predicted_path, "r", "utf-8").readlines())]
	
	gold_scores = [x[1] for x in golds]
	gold_min = min(gold_scores)
	gold_max = max(gold_scores)

	predict_scores = [x[1] for x in predicts]
	preds_min = min(predict_scores)
	preds_max = max(predict_scores)

	golds_norm = {x[0] : (x[1] - gold_min) / (gold_max - gold_min) for x in golds }
	preds_norm = {x[0] : (x[1] - preds_min) / (preds_max - preds_min) for x in predicts }
	preds_inv_norm = {key : 1.0 - preds_norm[key] for key in preds_norm}

	g_last = []
	p_last = []
	pinv_last = [] 
	for k in golds_norm:
		g_last.append(golds_norm[k])
		p_last.append(preds_norm[k])
		pinv_last.append(preds_inv_norm[k])

	pearson = stats.pearsonr(g_last, p_last)[0]
	spearman = stats.spearmanr(g_last, p_last)[0]
	pa = pairwise_accuracy(g_last, p_last)

	pearson_inv = stats.pearsonr(g_last, pinv_last)[0]
	spearman_inv = stats.spearmanr(g_last, pinv_last)[0]
	pa_inv = pairwise_accuracy(g_last, pinv_last)

	return max(pearson, pearson_inv), max(spearman, spearman_inv), max(pa, pa_inv)

gold_path = sys.argv[1]
pred_path = sys.argv[2]
pears, spear, pa = evaluate(gold_path, pred_path)
print("Pearson coefficient: " + str(pears))
print("Spearman coefficient: " + str(spear))
print("Pairwise accuracy: " + str(pa))