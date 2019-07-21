from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score,f1_score,average_precision_score,recall_score,precision_score,hamming_loss,precision_recall_curve
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.kernel_approximation import RBFSampler,Nystroem
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV,cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import numpy_indexed as npi

df = pd.read_csv('data.csv')
cvdf = pd.read_csv('sample_submission.csv')
df.drop(columns = ['Unnamed: 0'],inplace=True)

ok = list(df['game_season'].unique())
ok.remove(np.nan)
game_season_dict = dict(zip(ok,list(range(len(ok)))))
# game_season_dict
df['game_season'] = df['game_season'].apply(lambda x:game_season_dict[x] if x in game_season_dict.keys() else np.nan)

ok = list(df['area_of_shot'].unique())
ok.remove(np.nan)
area_of_shot_dict = dict(zip(ok,range(len(ok))))
df['area_of_shot'] = df['area_of_shot'].apply(lambda x:area_of_shot_dict[x] if x in area_of_shot_dict.keys() else np.nan)

ok = list(df['shot_basics'].unique())
ok.remove(np.nan)
shot_basics_dict = dict(zip(ok,range(len(ok))))
df['shot_basics'] = df['shot_basics'].apply(lambda x:shot_basics_dict[x] if x in shot_basics_dict.keys() else np.nan)

ok = list(df['range_of_shot'].unique())
ok.remove(np.nan)
range_of_shot_dict = dict(zip(ok,range(len(ok))))
df['range_of_shot'] = df['range_of_shot'].apply(lambda x:range_of_shot_dict[x] if x in range_of_shot_dict.keys() else np.nan)

df.drop(columns= ['team_name'],inplace=True)


ok = list(df['home/away'].unique())
ok.remove(np.nan)
home_away_dict = dict(zip(ok,range(len(ok))))
df['home/away'] = df['home/away'].apply(lambda x:home_away_dict[x] if x in home_away_dict.keys() else np.nan)

def split_lat_long(x):
    try:
        return list(map(float,x.split(',')))
    except:
        return [np.nan,np.nan]

df['lat/lng'].ffill(inplace=True)

df['lat']= df['lat/lng'].apply(lambda x: split_lat_long(x)[0])
df['lng']= df['lat/lng'].apply(lambda x: split_lat_long(x)[1])
latlongclf = KMeans(n_clusters=10)
df['lat/long'] = latlongclf.predict(df[['lat','lng']])

def combine_shots(x):
    try:
        x[0] = int(x[0][-2:])
    except:
        pass
    try:
        x[1] = int(x[1][-2:])
    except:
        pass
    if (np.isnan(x[0])) == False:
        return x[0]
    elif (np.isnan(x[1])) == False:
        return x[1]
    return np.nan
df['type_of_shot'] = df[['type_of_shot','type_of_combined_shot']].apply(combine_shots,axis=1)

def remaining_min(x):
    if np.isnan(x[0])==False:
        return x[0]
    else:
        return x[1]

df['remaining_min'] = df[['remaining_min','remaining_min.1']].apply(remaining_min,axis=1)

def power_of_shot(x):
    if np.isnan(x[0])==False:
        return x[0]
    else:
        return x[1]

df['power_of_shot'] = df[['power_of_shot','power_of_shot.1']].apply(power_of_shot,axis=1)

def knockout_match(x):
    if np.isnan(x[0])==False:
        return x[0]
    elif x[1]==0 or x[1]==1:
        return x[1]
    else:
        return x[0]

df['knockout_match'] = df[['knockout_match','knockout_match.1']].apply(knockout_match,axis=1)

def remaining_sec(x):
    if np.isnan(x[0])==False:
        return x[0]
    else:
        return x[1]

df['remaining_sec'] =  df[['remaining_sec','remaining_sec.1']].apply(remaining_sec,axis=1)

def distance_of_shot(x):
    if np.isnan(x[0])==False:
        return x[0]
    else:
        return x[1]

df['distance_of_shot'] = df[['distance_of_shot','distance_of_shot.1']].apply(distance_of_shot,axis=1)

df.drop(columns = ['lat/lng','type_of_combined_shot','remaining_min.1', 'power_of_shot.1', 'knockout_match.1',
       'remaining_sec.1', 'distance_of_shot.1', 'lat', 'lng'],inplace=True)

df['game_season'].ffill(inplace=True)
df['home/away'].ffill(inplace=True)
df.drop(columns = ['date_of_game'],inplace=True)

svmrbfclf = svm.SVC(kernel='rbf',probability =True,gamma='auto')
df.drop(columns = ['match_event_id'],inplace=True)

df.set_index('shot_id_number',inplace=True)
traindf = df.dropna(how='any')
y = traindf['is_goal']

a = list(traindf.columns)
a.remove('is_goal')
a.remove('team_id')
x = traindf[a]
for i in x.columns:
    x[i] = x[i]/x[i].max()

# x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3)
svmrbfclf.fit(x,y)

# predictionsrbf = svmrbfclf.predict(x_test)
output_df = df.loc[cvdf['shot_id_number']]
output_df.ffill(inplace=True)

for i in output_df.columns:
    output_df[i] = output_df[i]/traindf[i].max()

output_df['is_goal'] = svmrbfclf.predict_proba(output_df[a])[:,1]

output_df = output_df.reset_index()

cvdf['is_goal'] = output_df['is_goal']
cvdf.to_csv('result.csv')

print(cvdf)