#!/usr/bin/env python3
"""Reproduce the current unprojected metrics and t-SNE sensitivity table."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE, trustworthiness

parser=argparse.ArgumentParser()
parser.add_argument('--input',type=Path,default=Path(__file__).with_name('fig6_current_input.npz'))
parser.add_argument('--output',type=Path,default=Path('feature_audit_rerun'))
args=parser.parse_args()
args.output.mkdir(parents=True,exist_ok=True)
payload=np.load(args.input,allow_pickle=False)
raw=payload['features']
labels=payload['model_ids'].astype(str)
assert raw.shape==(927,6192) and np.isfinite(raw).all()
assert np.count_nonzero(raw[:,-1])==0
x=raw[:,:-1]
hist=x[:,6151:6181]
rows=[]
for name,arr,metric in [
 ('Original output features (6191)',x,'euclidean'),
 ('Original output features, cosine',x,'cosine'),
 ('Z-scored output features',StandardScaler().fit_transform(x),'euclidean'),
 ('Common neighbour histogram (30)',hist,'euclidean'),
 ('Z-scored neighbour histogram',StandardScaler().fit_transform(hist),'euclidean')]:
 rows.append(dict(space=name,dimensions=arr.shape[1],metric=metric,
                  silhouette=float(silhouette_score(arr,labels,metric=metric))))
pd.DataFrame(rows).to_csv(args.output/'fig6_current_original_space_metrics.csv',index=False)
sweep=[]
for p in [5,30,50]:
 for seed in [0,42,123]:
  y=TSNE(n_components=2,perplexity=p,random_state=seed,init='pca',max_iter=1000,n_jobs=1).fit_transform(x)
  sweep.append(dict(perplexity=p,seed=seed,silhouette=float(silhouette_score(y,labels)),
                    trustworthiness=float(trustworthiness(x,y,n_neighbors=5))))
pd.DataFrame(sweep).to_csv(args.output/'fig6_current_tsne_sensitivity.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False))
