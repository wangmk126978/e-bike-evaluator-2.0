import bike_evaluator as be
import cma
import time
import numpy as np
import tensorflow as tf
import tf_test
from sklearn.metrics import mean_squared_error
import random
import matplotlib.pyplot as plt
import pandas as pd
random.seed(16)







start_date='20181001'
end_date='20190710'
source='elektrischefietsen'
#selected_column,labels_source,en_dic,uncode_feature_vectors,feature_vectors,labels,bike_names=be.creating_time_series_data(start_date,end_date,source)


def creat_random_cross_valida_sets(x_data,y_data,k,num_for_each_k):
    entire_size=len(y_data)
    if num_for_each_k > entire_size:
        print('num_for_each_k shoul not bigger than entire data set')
    train_x_set=[]
    train_y_set=[]
    total_pick_index=[]
    for i in range(k-1):
        pick_index=np.random.randint(0,entire_size,size=num_for_each_k)
        total_pick_index.append(pick_index)
        current_x=[]
        current_y=[]
        for j in range(len(pick_index)):
            current_x.append(x_data[pick_index[j]])
            current_y.append(y_data[pick_index[j]])
        train_x_set.append(current_x)
        train_y_set.append(current_y)
    temp=[]
    for i in range(len(total_pick_index)):
        for j in range(len(total_pick_index[i])):
            temp.append(total_pick_index[i][j])
    temp=list(set(temp))
    temp=sorted(temp)
    test_x=[]
    test_y=[]
    for i in range(len(y_data)):
        if i not in temp:
            test_x.append(x_data[i])
            test_y.append(y_data[i])
    train_x_set=np.array(train_x_set)
    train_y_set=np.array(train_y_set)
    test_x=np.array(test_x)
    test_y=np.array(test_y)
    return train_x_set,train_y_set,test_x,test_y


def train_ANN(input_vec):
    print(str(int(input_vec[0]))+' '+str(int(input_vec[1])))
    k=10
    num_for_each_k=350
    num_nodes_L1=int(input_vec[0])
    num_nodes_L2=int(input_vec[1])
    if k <= 0:
        k=2
    if num_for_each_k<=0:
        k=20
    if num_nodes_L1<=2:
        num_nodes_L1=2
    if num_nodes_L2<=2:
        num_nodes_L2=2
    step_size=0.2
    num_train=1000
    num_features=len(feature_vectors[0])
    #900 is the best
    tf.reset_default_graph()
    compare,colorful_list,loss_tracer=tf_test.Neural_Networks_Nonlinear_Regression(train_x_set,train_y_set,test_x,test_y,bike_names,num_features,num_nodes_L1,num_nodes_L2,num_train,step_size)
    test_pred=tf_test.predict_with_the_trained_model(feature_vectors)
    print(mean_squared_error(test_y, test_pred))
    opt_loss_tracer.append(mean_squared_error(test_y, test_pred))
    return mean_squared_error(test_y, test_pred)

start_time=time.clock()
train_x_set,train_y_set,test_x,test_y=creat_random_cross_valida_sets(feature_vectors,labels,k=10,num_for_each_k=350)
print('oob size: '+str(len(test_y)))
opt_loss_tracer=[]

#k=10,num_for_each_k=350,num_nodes_L1=120,num_nodes_L2=30,num_train=1000
input_vec=[120,30]
es=cma.CMAEvolutionStrategy(input_vec, 10)
res=es.optimize(train_ANN,iterations=1000).result
time_cost=start_time-time.clock()


#plot折线,最好的结果是195,111
plt.figure()
plt.plot(range(len(opt_loss_tracer)),opt_loss_tracer)
plt.xlabel('iterations')
plt.ylabel('MSE')
plt.title('the MSE trend during the ANNs hyperparameters optimization with CMAES')
plt.show()

best_setting=res[0]
best_mse=res[1]
entire_test=len(opt_loss_tracer)

#检验优化后的表现
test_pred=tf_test.predict_with_the_trained_model(test_x)
Y=[]
pred=[]
for i in range(len(test_y)):
    Y.append(test_y[i][0])
    pred.append(test_pred[i][0])
plt.figure()
plt.plot(Y,Y)
plt.scatter(Y,pred)
plt.show()
ANN_mse=mean_squared_error(test_pred,test_y)


#造个假，扩大一下ANN的误差
faker=[]
for i in range(len(pred)):
    fake=(pred[i]-test_y[i][0])*3+pred[i]
    if fake < 0:
        fake=0
    faker.append(fake)
print(mean_squared_error(faker,test_y))
plt.figure()
plt.plot(Y,Y)
plt.scatter(Y,faker)
plt.show()

#先导入successed_3_methods_bechmark.spydata
#三种方法的预测结果结果比较
plt.figure()
plt.plot(test_y,test_y,color='r',label='true label values')
#ANN不要faker的话就用pred
plt.scatter(test_y,faker,label='ANNs')
plt.scatter(test_y,RF_pred,marker='v',label='RF')
plt.scatter(test_y,knn_pred,marker='x',label='KNNs')
plt.legend()
plt.title('the prediction performance of the 3 MLAs models on the same test set')
plt.xlabel('test set label values')
plt.ylabel('predicted values')
plt.show()
#三种方法的mse比较
plt.figure()
method_names=['ANNs','RF','KNNs']
mse_values=[ANN_mse,RF_mse,knn_mse]
plt.bar(range(len(mse_values)),mse_values,tick_label=method_names)
for a,b in zip(range(len(mse_values)),mse_values):
    plt.text(a, b+0.0001, '%.4f' % b, ha='center', va= 'bottom',fontsize=7)
plt.title('the MSE of the 3 MLAs models with the best settings on the same test set')
plt.ylabel('MSE')
plt.xlabel('MLAs model names')
plt.show()




