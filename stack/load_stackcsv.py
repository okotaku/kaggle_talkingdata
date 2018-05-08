import pandas as pd


def load_csv(df, path_list, predictors=None):
    for i, p in enumerate(path_list):
        fname = "stack{}".format(i)
        #df[fname] = pd.read_csv(p)['is_attributed'].values.astype('float16')
        rank = pd.read_csv(p)['is_attributed'].rank()
        rank /= len(rank)
        df[fname+"_rank"] = rank.astype('float16')
        if predictors is not None:
            #predictors.extend([fname, fname+"_rank"])
            predictors.append(fname+"_rank")
    return df, predictors