# -*- coding: utf-8 -*-
"""
@author: Lani QIU
@time: 20/1/2023 7:48 pm
1. spearmanr score by word & by feature
2. output top & bottom words in terms of MAE/ MSE
3. spearmanr cor between MAE (or MSE) and freq
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pathlib import Path

from collections import OrderedDict

from common.setup import *
from common.io_utils import general_reader, general_writer


def spr_words_feas(Y_test, Y_pred):
    """
    spearman cof across words & features
    @param Y_test: array (n_samples, n_features)
    @param Y_pred: array (n_samples, n_features)
    @return:
    """
    sp_w, sp_f = [], []
    if Y_test.shape != Y_pred.shape:
        assert False, "The size of the prediction array Y and of the test array Y are different."

    wn, fn = Y_test.shape
    for i in range(fn):

        var = spearmanr(Y_test[:, i], Y_pred[:, i])[0]
        sp_f.append(var)

    for i in range(wn):
        var = spearmanr(Y_test[i], Y_pred[i])[0]
        sp_w.append(var)

    return np.array(sp_f, dtype=float), np.array(sp_w, dtype=float)


def vis(arr, words, num):
    ss = np.argsort(arr)
    tt = zip(ss[:num], ss[-num:][::-1])
    out = []
    for i1, i2 in tt:
        tw, bw = words[i1], words[i2]  # list类型
        tp, bt = arr[i1], arr[i2]
        ll = map(str, [tw[0], tw[1], tp, bw[0], bw[1], bt])
        line = "\t".join(ll)
        out.append(line)
    return out


def main(fpth, in_dir, out_dir, gpat="gold", ppat="predict", num=10):
    """
    @param fpth: path to the ratings
    @param in_dir:
    @param out_dir: output dir
    @param gpat:
    @param ppat:
    @param num: visualize the top(bottom) num
    @return:
    """

    in_dir, out_dir = Path(in_dir), Path(out_dir)
    if not out_dir.exists():
        out_dir.mkdir()

    pths = in_dir.glob("*{}.npy".format(gpat))

    df = pd.read_excel(fpth)
    out1 = ["Model\tRegressor\tWord Correlation\tFeature Correlation\tMAE-Freq\tMSE-Freq\n"]
    out2 = ["Model\tRegressor\tTop(MAE)\t\t\tBottom(MAE)\t\t\tTop(MSE)\t\t\tBottom(MSE)\t\t\n"]

    for gpth in sorted(pths):
        logging.info("Processing {}".format(gpth))
        # load saved output & gt
        model, reg, _ = gpth.name.split("_")  # language model, regressor
        ppth = gpth.parent.joinpath(gpth.name.replace(gpat, ppat))
        gt = np.load(gpth).squeeze()
        pred = np.load(ppth).squeeze()
        sp_f, sp_w = spr_words_feas(gt, pred)  # spearmanr by word & by fea

        # load freq info
        wpth = in_dir.joinpath("{}_words.txt".format(model))
        words = [e.strip().split("\t") for e in general_reader(wpth)]
        fs = [f for _, f in enumerate(df["BCC(log10)"]) if [df["EngWords"][_], df["words"][_]] in words]  # freq
        fs = np.array(fs, dtype=float)

        # feat level
        mae = mean_absolute_error(gt, pred, multioutput="raw_values")
        mse = mean_squared_error(gt, pred, multioutput="raw_values")
        # word level
        maet = mean_absolute_error(gt.T, pred.T, multioutput="raw_values")
        mset = mean_squared_error(gt.T, pred.T, multioutput="raw_values")
        # 不同regressor的maet、mset

        spr_a = spearmanr(maet, fs)[0]
        spr_s = spearmanr(mset, fs)[0]

        ll = map(str, [model, reg, sp_w.mean(), sp_f.mean(), spr_a, spr_s])
        line = "\t".join(ll) + "\n"
        out1.append(line)

        va = vis(maet, words, num)
        vs = vis(mset, words, num)
        for idx, ach in enumerate(va):
            sch = vs[idx]
            out2.append("{}\t{}\t{}\t{}\n".format(model, reg, ach, sch))
    fout1 = out_dir.joinpath("cor_fea.txt")
    fout2 = out_dir.joinpath("cor_freq.txt")
    general_writer(out1, fout1)
    general_writer(out2, fout2)


def grouping(fpth, in_dir, out_dir, gpat="gold", ppat="predict", num=10):
    in_dir, out_dir = Path(in_dir), Path(out_dir)
    if not out_dir.exists():
        out_dir.mkdir()
    pths = in_dir.glob("*{}.npy".format(gpat))

    df = pd.read_excel(fpth)
    # 按type(或dimension)和POS分类
    ddict = {}
    for _, d in enumerate(df.iloc[4, 11:]):
        ddict[d] = ddict.get(d, []) + [_]
    pdict = dict(zip(df.EngWords[5:], df.pos[5:]))

    out1 = ["Group\tModel\tRegressor\tCorrelation\n"]
    out2 = ["Group\tModel\tRegressor\tCorrelation\n"]

    for gpth in sorted(pths):
        logging.info("Processing {}".format(gpth))
        # load saved output & gt
        model, reg, _ = gpth.name.split("_")  # language model, regressor

        ppth = gpth.parent.joinpath(gpth.name.replace(gpat, ppat))
        gt = np.load(gpth).squeeze()
        pred = np.load(ppth).squeeze()

        wpth = in_dir.joinpath("{}_words.txt".format(model))
        words = [e.strip().split("\t") for e in general_reader(wpth)]

        if "align" in model:  # todo 跳过align
            continue
        model = mapping[model]

        # vec按pos grouping
        pos_gp = {}
        for _, (e, c) in enumerate(words):
            pos = pdict[e]
            pos_gp[pos] = pos_gp.get(pos, []) + [_]
        for ky, va in pos_gp.items():
            sp_f, sp_w = spr_words_feas(gt[va], pred[va])
            # word level
            maet = mean_absolute_error(gt[va].T, pred[va].T, multioutput="raw_values")
            mset = mean_squared_error(gt[va].T, pred[va].T, multioutput="raw_values")
            ll = map(str, [ky, model, reg, sp_w.mean()])
            line = "\t".join(ll) + "\n"
            out1.append(line)

        for ky, va in ddict.items():
      
            # spearmanr(Y_test[i], Y_pred[i])[0]
            sp_f, sp_w = spr_words_feas(gt.T[va], pred.T[va])


            # maet = mean_absolute_error(gt[va], pred[va], multioutput="raw_values")
            # mset = mean_squared_error(gt[va], pred[va], multioutput="raw_values")
            ll = map(str, [ky, model, reg, sp_w.mean()])
            line = "\t".join(ll) + "\n"
            out2.append(line)

        fout1 = out_dir.joinpath("words_grouping.txt")
        fout2 = out_dir.joinpath("feat_grouping.txt")
        general_writer(out1, fout1)
        general_writer(out2, fout2)




def draw_heatmap(fpth, fout, fig=(3, 5)):
    # 绘图
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=fig, dpi=200)
    sns.set(font_scale=1)
    # 构造dataframe
    ddict = {}
    rnames = []
    for line in general_reader(fpth)[1:]:
        gr, emb, _, wc = line.strip().split("\t")
        if "align" in emb:
            continue
        ddict[gr] = ddict.get(gr, []) + [float(wc)]

        if emb not in rnames:
            rnames.append(emb)
    # names = []

    # for i in rnames:
    #     names.append(mapping[i])
    dr = pd.DataFrame.from_dict(ddict, )
    dr.index = rnames
    ax = sns.heatmap(dr, cmap="YlOrRd", xticklabels=True, yticklabels=True,)
    plt.tick_params(axis="both", labelsize=10)
    plt.savefig(fout)
    plt.show()




if __name__ == '__main__':

    _ddir = adr.joinpath("binder")

    fpth = _ddir.joinpath("new_ratings.xlsx")
    out_dir = _ddir.joinpath("out_grouping")
    tmp_dir = _ddir.joinpath("out_grouping")

    # main(fpth, tmp_dir, out_dir)
    mapping = {"cc.zh.300":  "fast.cc.zh",
               "sgns.wiki": "sgns.wiki.zh",
               "wiki.zh": "fast.wiki.zh"}

    # grouping(fpth, tmp_dir, out_dir)
    draw_heatmap(out_dir.joinpath("words_grouping.txt"),
                 out_dir.joinpath("pos.png"),
                 (3,4))















