import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate(df, seed=42):
    if df.empty or df.event_id.nunique()<4: raise ValueError("At least 4 events required for held-out-event evaluation")
    features=[c for c in df.columns if c not in {"event_id","rsn_1","rsn_2","delta_orientation_deg"}]
    splitter=GroupShuffleSplit(n_splits=1,test_size=.25,random_state=seed)
    tr,te=next(splitter.split(df,groups=df.event_id))
    train,test=df.iloc[tr],df.iloc[te]
    if len(train)<10 or len(test)<5: raise ValueError("Too few train/test pairs; gather more independent events")
    Xtr,Xte=train[features],test[features]
    ytr,yte=train.delta_orientation_deg,test.delta_orientation_deg
    models={"Distance-only Ridge":make_pipeline(SimpleImputer(strategy="median",keep_empty_features=True),StandardScaler(),Ridge(alpha=10)),
            "Ridge":make_pipeline(SimpleImputer(strategy="median",keep_empty_features=True),StandardScaler(),Ridge(alpha=10)),
            "MLP":make_pipeline(SimpleImputer(strategy="median",keep_empty_features=True),StandardScaler(),MLPRegressor(hidden_layer_sizes=(32,16),alpha=10,early_stopping=True,max_iter=500,random_state=seed))}
    metrics=[]; preds=[]
    for name, model in models.items():
        columns=["station_separation_km"] if name=="Distance-only Ridge" else features
        model.fit(Xtr[columns],ytr)
        yp=np.clip(model.predict(Xte[columns]),0,90)
        metrics.append({"model":name,"test_mae_deg":mean_absolute_error(yte,yp),
                        "test_rmse_deg":np.sqrt(mean_squared_error(yte,yp)),
                        "test_r2":r2_score(yte,yp),"train_pairs":len(tr),"test_pairs":len(te),
                        "train_events":train.event_id.nunique(),"test_events":test.event_id.nunique()})
        preds.extend({"model":name,"event_id":e,"actual":a,"predicted":p} for e,a,p in zip(test.event_id,yte,yp))
    return pd.DataFrame(metrics),pd.DataFrame(preds)
