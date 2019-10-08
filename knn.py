import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cma
import bike_evaluator as be
from sklearn.metrics import mean_squared_error
 
#要分训练集和测试集才行
train_x_set,train_y_set,test_x,test_y=be.creat_random_cross_valida_sets(feature_vectors,labels,k=5,num_for_each_k=700)

def predict_y(k,input_vec,feature_vectors,labels):
    dist_mat=[]
    for i in range(len(feature_vectors)):
        dist_mat.append([i,np.sqrt(np.sum(np.square(feature_vectors[i] - input_vec)))])
    dist_mat=sorted(dist_mat, key=lambda x:x[1],reverse = False)
    knn=[]
    output=0
    #需要加权,knn收集所有的邻居的距离，然后将这些邻居的距离正则化，使相加等于1，然后再计算带权重的output
    for i in range(k):
       knn.append(labels[dist_mat[i][0]][0])
    sum_knn_dist=sum(knn)
    for i in range(k):
       output+=(knn[i]/sum_knn_dist)*labels[dist_mat[i][0]][0]
    output/=k
    return knn,output

def calculate_mse(y_pred,y_test):
    sqrt_error=0
    for i in range(len(y_pred)):
        sqrt_error+=pow((y_pred[i]-y_test[i][0]),2)
    mse=sqrt_error/len(y_pred)
    return mse

def plot_figure(y_pred,y_test):
    plt.figure()
    plt.plot(y_test,y_test,'b-',label = 'real score')
    plt.scatter(y_test,y_pred,marker='x',color='g',label = 'predict score',alpha=0.6)
    plt.title('The relationship between the real_score and the predict_score')
    plt.xlabel('y labels')       #x轴的标签
    plt.ylabel('y predict')       #y轴的标签
    plt.legend()            #设置图例
    plt.show()

#(错的，不要理这个函数)找到最合适的k
def test_k(k):
    k=int(k[0])
    if k<=2:
        k=2
    if k>=700:
        k=700
    ave_mse=0
    for i in range(len(train_x_set)):
        y_pred=[]
        for j in range(len(test_x)):
            knn,output=predict_y(k,test_x[j],train_x_set[i])
            y_pred.append(output)
        mse=calculate_mse(y_pred,test_y)
        ave_mse+=mse
    ave_mse/=len(train_x_set)
    print('k='+str(k)+', ave_mse:'+str(ave_mse))
    plot_figure(y_pred,test_y)
    mse_tracer.append(ave_mse)
    return ave_mse

k=[7,2]
ftarget=0.013
mse_tracer=[]
#k_mse=test_k(k)
res=cma.fmin(test_k,k,1,options={'ftarget':ftarget,'popsize':10})
        
#遍历所有的k，找的最合适的k
def k_tester(k,test_x,test_y,removed_train_x,removed_train_y):
    y_pred=[]
    for i in range(len(test_x)):
        knn,output=predict_y(k,test_x[i],removed_train_x,removed_train_y)
        y_pred.append(output)
    mse=mean_squared_error(y_pred,test_y)
    print('k='+str(k)+', mse:'+str(mse))
    return mse

mse_tracer=[]
for i in range(100):
    mse_tracer.append(['k = '+str(i+1),k_tester(i+1,test_x,test_y,removed_train_x,removed_train_y)])
    
del mse_tracer[0]
#plot mse随k变化的曲线,证明1是最好的
Y=[]
for i in range(len(mse_tracer)):
    Y.append(mse_tracer[i][1])
plt.figure()
plt.plot(range(1,len(mse_tracer)+1),Y)
plt.xlabel('the number of the nearest neighbors(k)')
plt.ylabel('MSE')
plt.title('the MSE trend of the KNNs models during the growth of k')
plt.show()

#看knn在测试集上的表现
knn_pred=[]
for i in range(len(test_x)):
    knn,output=predict_y(1,test_x[i],removed_train_x,removed_train_y)
    knn_pred.append(output)  
plt.figure()
plt.plot(test_y,test_y)
plt.scatter(test_y,knn_pred)
plt.show()    

knn_mse=mean_squared_error(knn_pred,test_y)



 