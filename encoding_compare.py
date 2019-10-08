import bike_evaluator as be
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import numpy as np 




start_date='20181001'
end_date='20190710'
source='elektrischefietsen'
k=10
num_for_each_k=350

def test_performance(train_x,train_y,test_x,test_y,n):
    regr = RandomForestRegressor(n_estimators=n,oob_score=False,bootstrap=True)
    regr.fit(train_x, train_y)
    mse=mean_squared_error(regr.predict(test_x),test_y)
    return mse

def k_fold_cross_validation_T_times(k,X_set,Y_set,function,T,n):
    #运行t次
    t_times_mse=[]
    for t in range(T):
        #组装数据集
        data_set=[]
        for i in range(len(X_set)):
            #加个序号，方便不放回操作
            data_set.append([i])
            for j in range(len(X_set[i])):
                data_set[i].append(X_set[i][j])
            data_set[i].append(Y_set[i])
        #抽取k-1次,每次抽出int(len(data_set/k))个,抽取一次后重构一次data_set（因为是不放回的）
        fold_size=int(len(data_set)/k)
        k_folder=[]
        current_data_set=data_set
        for i in range(k-1):
            k_folder.append(random.sample(current_data_set,fold_size))
            new_data_set=[]
            for j in range(len(current_data_set)):
                for z in range(len(k_folder[i])):
                    if current_data_set[j][0] == k_folder[i][z][0]:
                        break
                    if z == len(k_folder[i])-1:
                        new_data_set.append(current_data_set[j])
            current_data_set=new_data_set
        k_folder.append(current_data_set)
        #移除序号
        for i in range(len(k_folder)):
            for j in range(len(k_folder[i])):
                del k_folder[i][j][0]
        #构建k次训练集和测试集并取平均mse
        mse_set=[]
        for i in range(k):
            test_index=i
            train_set=[]
            for j in range(len(k_folder)):
                if j == test_index:
                    test_set=k_folder[j]
                if j != test_index:
                    for z in range(len(k_folder[j])):
                        train_set.append(k_folder[j][z])
            #将x和y分离出来
            train_x=[]
            train_y=[]
            test_x=[]
            test_y=[]
            for j in range(len(train_set)):
                train_y.append(train_set[j][-1][0])
                train_x.append([])
                for z in range(len(train_set[j])-1):
                    train_x[j].append(train_set[j][z])
            for j in range(len(test_set)):
                test_y.append(test_set[j][-1][0])
                test_x.append([])
                for z in range(len(test_set[j])-1):
                    test_x[j].append(test_set[j][z])
            mse_set.append(function(train_x,train_y,test_x,test_y,n))
        avg_mse=np.mean(mse_set)
        t_times_mse.append(avg_mse)
    t_times_avg_mse=np.mean(t_times_mse)
    return t_times_avg_mse



#测试one-hot
selected_column,labels_source,en_dic,uncode_feature_vectors,feature_vectors,labels,bike_names=be.creating_time_series_data(start_date,end_date,source)
one_hot_mse=[]
for i in range(100):
    mse=k_fold_cross_validation_T_times(5,feature_vectors,labels,test_performance,5,i+1)
    one_hot_mse.append(mse)
    print('one hot, n = '+str(i+1)+', mse: '+str(mse))
    
#再配上一个散点图







selected_column,labels_source,en_dic,uncode_feature_vectors,feature_vectors,labels,bike_names=be.creating_time_series_data(start_date,end_date,source)
dummy_mse=[]
for i in range(100):
    mse=k_fold_cross_validation_T_times(5,feature_vectors,labels,test_performance,5,i+1)
    dummy_mse.append(mse)
    print('one hot, n = '+str(i+1)+', mse: '+str(mse))
np.mean(dummy_mse)
    
plt.figure()
plt.plot(list(range(100)),one_hot_mse,label='one-hot encoding')
plt.plot(list(range(100)),dummy_mse,label='dummy encoding')
plt.xlabel('number of decision trees(n)')
plt.ylabel('MSE')
#plt.title('The MSE performance comparison between one hot encoding and dummy encoding(s=270)')
plt.legend()
plt.show()
    
    
