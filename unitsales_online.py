# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import xgboost as xgb

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

stores = pd.read_csv("../input/stores.csv")
transactions = pd.read_csv("../input/transactions.csv")
items = pd.read_csv("../input/items.csv")
holiday = pd.read_csv("../input/holidays_events.csv")
oil = pd.read_csv("../input/oil.csv")
train_ori = pd.read_csv("../input/train.csv")
#train_1 = train_ori.sample(frac=0.8,axis=0,replace=False)
print("load data done")

# %% Define functions

def applyAtRow(X, along, metrics, aggr=[], days=None, assign_index=True) :
    ### apply at last N rows. For caculation of recent rows.
    rlt = pd.DataFrame([])
    for agg_func in aggr :
        agg_all = pd.DataFrame([])
        if days == None:
            days = X.shape[0]
        for r in X[along]:
            agg_row = pd.DataFrame([agg_func(X.loc[X[along]<=r, metrics].tail(days))])
            agg_all = pd.concat([agg_all, agg_row])
        if assign_index == True :
            agg_all.index = X.index
            agg_all.columns = [agg_func.__name__ + "_" + str(days) + "d"]
        rlt = pd.concat([rlt, agg_all], axis=1)
    return rlt

def applyAtDate(X, along, metrics, format, aggr=[], days=7, assign_index=True) :
    ### apply at given date, but not necessarily the last N rows.
    ### along:must be datetime or can be transformed to datetime, as for transforming, format is necessary.
    ##transtype of along column
    X[along] = pd.to_datetime(X[along], format=format)
    rlt = pd.DataFrame([])
    init = True;
    col_d = pd.DataFrame([])
    for agg_func in aggr :
        agg_all = pd.DataFrame([])
        for r in X[along]:
            lower_r = r -  pd.Timedelta(str(days)+" days")
            X1 = X.loc[X[along] <= r, :]
            X2 = X1.loc[X1[along] > lower_r, :]

            agg_row = pd.DataFrame({agg_func.__name__ + "_" + str(days) + "d" : [agg_func(X2.loc[:, metrics])]  })
#            agg_all.append(agg_row)
            agg_all = pd.concat([agg_all, agg_row],axis=0)
            if init == True :
                col_d = pd.concat([col_d, pd.DataFrame({"date": [r]})], axis=0)
#        if assign_index == True :
#            agg_all.index = X.index
 #            agg_all.columns = ["date",]
        if init == True :
            init = False
            rlt = pd.concat([rlt, col_d], axis=1)
        rlt = pd.concat([rlt, agg_all], axis=1)
    return rlt

def applyRecentNDay(g, days, agg_func=[]) :
#    if type(g) <> pd.
    agg = pd.DataFrame([])
    for i in g.groups :
        d = g.get_group(i)
        tmp = applyAtDate(d, "date", "transactions"
                         , "%Y-%m-%d", agg_func, days, assign_index=True)
        tmp.index = np.repeat(i, tmp.shape[0])
        agg = pd.concat([agg, tmp],axis=0)
    return agg

def genFeature(X) :

    features = pd.merge(X, store_features, how='left', on='store_nbr')
    print("Join store feature done")
    features = pd.merge(features, item_features, how='left', on='item_nbr')
    print("Join item feature done")
    features = pd.merge(features, oil_features, how='left', on='date')
    print("Join oil feature done")
    features = pd.merge(features, hol_features, how='left', on='date')
    print("Join hol feature done")
    # features.index = pd.MultiIndex.from_arrays([features["store_nbr"],features["date"]], names=["store_nbr","date_x"])
    features = pd.merge(features, txn_features, how='left', on=['store_nbr','date'])
    print("Join txn feature done")
    return features
 
print("create functions done")

# %% Store Features
###By city agg
bycity_cnt =  stores.groupby("city", as_index=False).aggregate({
                "store_nbr" : lambda x: x.nunique(),
                "type" : lambda x: x.nunique(),
                "cluster" : lambda x: x.nunique()}) \
      .rename(columns={"store_nbr": "city_store_cntd",
                       "type" : "city_type_cntd",
                       "cluster" : "city_cluster_cntd"})
bystate_cnt = stores.groupby("state", as_index=False).aggregate({
                "store_nbr" : lambda x: x.nunique(),
                "type" : lambda x: x.nunique(),
                "cluster" : lambda x: x.nunique()}) \
      .rename(columns={"store_nbr": "state_store_cntd",
                       "type" : "state_type_cntd",
                       "cluster" : "state_cluster_cntd"})
bytype_cnt = stores.groupby("type", as_index=False).aggregate({
                "store_nbr" : lambda x: x.nunique(),
                "city" : lambda x: x.nunique(),
                "cluster" : lambda x: x.nunique()}) \
      .rename(columns={"store_nbr": "type_store_cntd",
                       "city" : "type_city_cntd",
                       "cluster" : "type_cluster_cntd"})
bycluster_cnt = stores.groupby("cluster", as_index=False).aggregate({
                "store_nbr" : lambda x: x.nunique(),
                "city" : lambda x: x.nunique(),
                "type" : lambda x: x.nunique()}) \
      .rename(columns={"store_nbr": "cluster_store_cntd",
                       "type" : "cluster_type_cntd",
                       "city" : "cluster_city_cntd"})
# print(bycity_cnt.head(), bystate_cnt.head())
# Merge and get Store features
store_features = pd.merge(stores, bycity_cnt, how='left', on="city")
store_features = pd.merge(store_features, bystate_cnt, how='left', on="state")
store_features = pd.merge(store_features, bytype_cnt, how='left', on="type")
store_features = pd.merge(store_features, bycluster_cnt, how='left', on="cluster")
# print(store_features.head())
print("Store features Done")

#%% up-to-date sum/avg
tran_sp = transactions
g_store = tran_sp.groupby("store_nbr")
tran_7d = applyRecentNDay(g_store, 7, [np.sum, np.mean, np.std, lambda x : (np.max(x)-np.min(x))/len(x)]) \
            .rename(columns={"<lambda>_7d":"diff_7d"}).reset_index()
tran_30d = applyRecentNDay(g_store, 30, [np.sum, np.mean, np.std, lambda x : (np.max(x)-np.min(x))/len(x)]) \
            .rename(columns={"<lambda>_30d":"diff_30d"}).reset_index()
tran_365d = applyRecentNDay(g_store, 365, [np.sum, np.mean, np.std, lambda x : (np.max(x)-np.min(x))/len(x)]) \
            .rename(columns={"<lambda>_365d":"diff_365d"}).reset_index()
txn_features = pd.merge(tran_7d, tran_30d, on=["index","date"], how='inner')
txn_features = pd.merge(txn_features, tran_365d, on=["index","date"], how='inner')
print("txn_features Done")

# %% Item Features
###By city agg
byfamily_cnt =  items.groupby("family", as_index=False).aggregate({
                "item_nbr" : lambda x: x.nunique(),
                "class" : lambda x: x.nunique(),
                "perishable" : np.sum}) \
      .rename(columns={"item_nbr": "family_item_cntd",
                       "class" : "family_class_cntd",
                       "perishable" : "family_perish_cntd"})
byclass_cnt = items.groupby("class", as_index=False).aggregate({
                "item_nbr" : lambda x: x.nunique(),
                "perishable" : np.sum }) \
      .rename(columns={"item_nbr": "class_item_cntd",
                       "perishable" : "class_perish_cntd"})
# %% Item Features
item_features = pd.merge(items, byfamily_cnt, how='left', on="family")
item_features = pd.merge(item_features, byclass_cnt, how='left', on="class")
print("item_features Done")

# print(item_features.head())
# %% Date Features
# %%Oil Price
oil_full = applyAtRow(oil, "date", "dcoilwtico", aggr=[np.sum, np.mean, np.std])
oil_7d   = applyAtRow(oil, "date", "dcoilwtico", aggr=[np.sum, np.mean, np.std], days=7)
oil_30d  = applyAtRow(oil, "date", "dcoilwtico", aggr=[np.sum, np.mean, np.std], days=30)
oil_365d = applyAtRow(oil, "date", "dcoilwtico", aggr=[np.sum, np.mean, np.std], days=365)
oil_features = pd.concat([oil,oil_full,oil_7d,oil_30d,oil_365d],axis=1)
print("oil features done")

###Holidays
hol_7d  = applyAtDate(holiday, "date", "type", "%Y-%m-%d", aggr=[len,lambda x: len(np.unique(x))], days=7)
hol_30d = applyAtDate(holiday, "date", "type", "%Y-%m-%d", aggr=[len,lambda x: len(np.unique(x))], days=30)
hol_90d = applyAtDate(holiday, "date", "type", "%Y-%m-%d", aggr=[len,lambda x: len(np.unique(x))], days=90)
#print(hol_7d.shape, hol_30d.shape, hol_90d.shape)
hol_features = pd.concat([hol_7d
                         ,hol_30d.drop(["date"],axis=1)
                         ,hol_90d.drop(["date"],axis=1)], axis=1) \
                .rename(columns = {"<lambda>_7d" : "holtype_cntd_7d"
                                   ,"<lambda>_30d" : "holtype_cntd_30d"
                                   ,"<lambda>_90d" : "holtype_cntd_90d"})
print("holiday done")
######
##feature checkpoint
######
store_features.to_csv("store_features.csv", index=False)
item_features.to_csv("item_features.csv", index=False)
oil_features.to_csv("oil_features.csv", index=False)
hol_features.to_csv("hol_features.csv", index=False)
txn_features.to_csv("txn_features.csv", index=False)
print("feature checkpoint done")
 
####generate feature
features = genFeature(train_ori.iloc[125367040:,:])
print("features done")




# %% Try Models
# %%
X_train = features.drop(["unit_sales","date_y","date_x", "city", "state", "type", "family"],axis=1)
Y_train = features["unit_sales"]

test_features = genFeature(test)
# %% 
X_test = test_features.drop(["date_y","date_x", "city", "state", "type", "family"],axis=1)
print("dataset done")

dtest = xgb.DMatrix(X_test)
# %%
#X_train[["date_x", "city", "state", "type", "family"]] = X_train[["date_x", "city", "state", "type", "family"]]  \
#X_train[["city"]] = X_train[["city"]].apply(lambda x: pd.factorize(x.astype("category")) , axis=0)

dtrain = xgb.DMatrix(X_train,  Y_train)

print("DMatrix done")

param = {'max_depth':2, 'eta':0.5, 'silent':1, 'objective':'reg:linear' }
num_round = 3000
bst = xgb.train(param, dtrain, num_round)

print("model fitting done")

pred = bst.predict(dtest)

print("predict done")

df_id = (X_test["id"]).reset_index()
df_pred = pd.DataFrame({"unit_sales":pred})
df_out = pd.concat([df_id["id"],df_pred],axis=1)
print("df_out done")
df_out.to_csv("../pred.csv", index=False)
print("output done")

# Any results you write to the current directory are saved as output.