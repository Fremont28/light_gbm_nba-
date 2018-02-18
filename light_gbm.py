nba_data=pd.read_csv("NBA draft class.csv")
nba_data.info() 
nba_data1=nba_data.dropna(subset=['VORP>0'])
#subset data and convert to numpy array 
X=nba_data1.iloc[:,[12,13,14,15,16,17]].values 
y=nba_data1.iloc[:,6:7].values 
#split the data into test and train
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.31,random_state=0)
#feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler() 
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test) 
#model building 
#convert training data into lightgbm format
import lightgbm as lgb
nba_d_train = lgb.Dataset(X_train, label=y_train)

params={}
params['learning_rate']=0.003
params['boosting_type']='gbdt'
params['objective']='binary'
params['metric']='binary_logloss'
params['sub_feature']=0.5
params['num_leaves']=10
params['min_data']=50
params['max_depth']=10

clf=lgb.train(params,nba_d_train,50) #train model with 50 iterations 
#predictions 
y_pred=clf.predict(X_test)
#convert into binary values
for i in range(0,50):
   if y_pred[i]>0.5:
       y_pred[i]=1
    else:
        y_pred[i]=0
#check accuracy (confusion matrix)
con_mat=confusion_matrix(y_test,y_pred)