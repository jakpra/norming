
import pandas as pd
import logging

from mi import cal_pairwise_mi, cal_bt_attrs, generate_heatmap


def main(ratings, attr_names, sorted_attrs, spth, handle_nans="pad"):
    """
    @param ratings: pandas series
    @param attr_names:  list
    @param sorted_attrs: list
    @param spth: path to save the heatmap
    @param handle_nans: how to deal with nan values
    @return:
    """
    n = ratings.shape[1]
    logging.info("compute pairwise mi")
    mis = cal_bt_attrs(n, ratings, method=handle_nans)
    # groups = df[1]['Group'].dropna().values.tolist()
    # grp_attr = df[1]["Group-Attr"].dropna().tolist()
    # logging.info("compute in-group pairwise mi")
    # gr_scores = cal_pairwise_mi(groups, grp_attr, attr_names, mis)
    logging.info("draw heatmap")
    # generate_heatmap(mis.copy(), n, attr_names, sorted_attrs, spth)


if __name__ == '__main__':
    fpth = "/Users/laniqiu/My Drive/dough/Copy of meanRating_July1.xlsx"

    spth = ""
    df = pd.read_excel(fpth)
    ratings = df.iloc[5:, 11:]
    attr_names = df.iloc[3, 11:].to_list()
    sorted_attrs = attr_names
    spth = "/Users/laniqiu/My Drive/dough/mi_hsu.png"
    main(ratings, attr_names, sorted_attrs, spth)

