#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# # -*- coding: utf-8 -*-
# from flask import Flask, jsonify, request, render_template
import pandas as pd

pod=pd.read_csv("document.csv")

pod["docno"]=pod.docno.astype(str)
pod["title"]=pod.title.astype(str)
pod["description"]=pod.description.astype(str)
pod["genre"]=pod.genre.astype(str)
pod["rating"]=pod.rating.astype(str)
pod["num_reviews"]=pod.num_reviews.astype(str)

import pyterrier as pt
import pandas as pd
import os
if not pt.started():
    pt.init()

pd_indexer = pt.DFIndexer("./pd_index", overwrite=True, blocks=True)
indexref = pd_indexer.index(pod["description"], pod["docno"],pod["title"],pod["genre"],pod["rating"],pod["num_reviews"])

index = pt.IndexFactory.of(indexref)

topics = pd.read_csv('./topics.csv')
topics['qid'] = topics.qid.astype('str')
qrels = pd.read_csv('./qrels.csv')
qrels['qid'] = qrels.qid.astype(str)
qrels['docno'] = qrels.docno.astype(str)

bm25 = pt.BatchRetrieve(index, wmodel="BM25")
tf_idf = pt.BatchRetrieve(index, wmodel="TF_IDF")

RANK_CUTOFF = 30
SEED=42

from sklearn.model_selection import train_test_split

tr_va_topics, test_topics = train_test_split(topics, test_size=0.3, random_state=SEED)
train_topics, valid_topics = train_test_split(tr_va_topics, test_size=0.1, random_state=SEED)

tf_idf = pt.BatchRetrieve(index, wmodel="TF_IDF")
bm25 = pt.BatchRetrieve(index, wmodel="BM25")
qe = pt.rewrite.Bo1QueryExpansion(index)


ltr_feats = (bm25%RANK_CUTOFF) >> pt.text.get_text(index, ["genre","title","rating","num_reviews"]) >> (
    pt.transformer.IdentityTransformer()
    **
    (bm25>>qe>>bm25)
    **
    (pt.text.scorer(body_attr="genre", wmodel='BM25') )
    **
    (pt.text.scorer(body_attr="title", wmodel='BM25') )
    **
    pt.apply.doc_score(lambda row: float(row["rating"]))
    **
    pt.apply.doc_score(lambda row: int(row["num_reviews"]))
)


import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
xg = xgb.XGBRegressor(max_depth = 20, learning_rate = 0.01, n_estimators = 100, objective = 'rank:pairwise')#rank:pairwise rank:ndcg
xg_pipe = ltr_feats >> pt.ltr.apply_learned_model(xg)
xg_pipe.fit(train_topics, qrels)
detail=pd.read_csv("detailed_info.csv")

from flask import Flask, jsonify, request, render_template
app = Flask(__name__)
@app.route('/')
def hello():
    return render_template("index.html")

@app.route('/index', methods = ['POST'])
def index():
    sentence = request.form['sentence']
    try:
        df = detail.iloc[xg_pipe(sentence).sort_values(by=['rank']).docid]
        n = min(len(df), 5)
        res = ''
        for i in range(n):
            res += 'Podcast'+ str(i+1) + ':' + '\n' + df['title'].iloc[i] + '\n' + df['link'].iloc[i] + '\n' + '\n'
    except:
        res = 'No Results. Please change a search word.'
    print(res)
    return jsonify({'sentence': res})

if __name__=='__main__':
    app.run()


