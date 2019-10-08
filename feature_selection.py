"""
把vendorlink写出来
写完feature selection只留图
其他几个用RF的结果要重新跑
三种feature方法一种一个图
最后一个表，可以撑三页还要多
全部结果跑出来，补完所有缺口，晚上回去增加篇幅

1. 尝试了case base的方法，就是picking the most similar bikes for regression,但是效果没有我们之前用的两种方法好.

2. 完善了engine brand的分析， 但是brakes' information is not complete，目前有more than 600 e-bike models we haven't 
brake information, 我正在手动搜索information for the missing brake data.
################################
1. time series data analysis
2. feature reducing analysis
3. communicated with BI，to see if they have any additional suggestions,and save the useful models，
4. help the finance group to update vendorlinks again.
##############################

画个feature selection的示意图

wrapper耗时过长，embedded耗时短，最终找到最好的表现，但是我们不选这个表现，我们自己选出来的跟追好的比一比

ANN的sigmoid和tanh的表现区别

conclusion：
1.文章总结：总的来说MLAs可以用于商业分析，并且网页流量数据的用户行为分析比问卷调查更高效可靠，在电动自行车的time series
 analysis中我们发现，不同time span对于models的pattern learning还是存在巨大影响，所以即使有了可靠的data source，还是
有必要进行time series的分析，来确定最合适的time span。 其次我们发现其实components encoding方式其实对models的performance
影响并不大，尤其是RF，在n比较小时合适的encoding method会导致更好的表现，但当决策树数量变大后，不同的编码方式其实不会对结果
造成巨大的影响。挨着写...
在我们所用的MLAs中

2.对于结果的讨论，什么情况下用RF什么情况下用ANN, CMAES在调整MLAs地参数时是非常好用的，
3.可以实现什么应用，在其他行业中扩展的可能性（只要选好了labels source和features source甚至可以自动完成automatic machine learning training）


4. limitation 是什么
5. 以后可以怎么改进
5.1 可以扩充数据库
5.2 可以改进网页结构使得labels source更加可靠，
5.3 可以增加features 的attributes class，并选择出更有效的features
5.4 如果要用supervised learning必须要重视labels的质量，需要更加规范和完整地记录信息
5.5 因为网站上的自行车展示图片规则差异巨大，所以如果自行车的展示规则一致的话，我们甚至可以直接用图片生成外观的features





检查：
1. “paper” 都要改成 “thesis”
2. 谷歌检查所有翻译
3. 检查公式的逻辑和表达
4. 统一某些描述词汇
5. 可数不可数检查
6. 图片名字改写

"""
import bike_evaluator as be
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import cma
import time
import matplotlib.pyplot as plt
import random
from scipy.special import comb, perm

#计算所有的可能性
num_of_possible_combination=0
for i in range(37):
    num_of_possible_combination+=comb(37,i+1)

e_bike_comp = pd.read_csv('e_bike_excel_copy.csv',sep=';')
e_bike_comp=e_bike_comp.values
e_bike_comp=e_bike_comp.tolist()

def must_not_choose(selected_column):
    selected_column[be.alpha2num('a')]=0
    selected_column[be.alpha2num('d')]=0
    selected_column[be.alpha2num('k')]=0
    selected_column[be.alpha2num('q')]=0
    selected_column[be.alpha2num('ae')]=0
    selected_column[be.alpha2num('ag')]=0
    selected_column[be.alpha2num('ah')]=0
    return selected_column

def merge_train_set():
    k=10
    num_for_each_k=350
    train_x_set,train_y_set,test_x,test_y=be.creat_random_cross_valida_sets(feature_vectors,labels,k,num_for_each_k)
    train_x=[]
    train_y=[]
    for i in range(len(train_x_set)):
        for j in range(len(train_x_set[i])):
            train_x.append(train_x_set[i][j])
            train_y.append(train_y_set[i][j])
    return train_x,train_y,test_x,test_y
    
def wrapper_feature_selection(selected_column):
    must_not_choose(selected_column)
    encoder=[]
    for i in range(len(selected_column)):
        if selected_column[i]>=1:
            encoder.append(be.num2alpha(i))
    selected_column=encoder
    print(selected_column)
    uncode_feature_vectors,feature_vectors,labels,bike_names,en_dic=be.generate_features_and_labels(labels_source,selected_column,e_bike_comp)
    train_x,train_y,test_x,test_y=merge_train_set()
    mse=[]
    for i in range(5):
        regr = RandomForestRegressor(n_estimators=200,oob_score=False,bootstrap=True)
        regr.fit(train_x, train_y)
        mse.append(mean_squared_error(regr.predict(test_x),test_y))
        print(str(i)+' train the mse is: '+str(mean_squared_error(regr.predict(test_x),test_y)))
    opt_loss_tracer.append(np.mean(mse))
    return np.mean(mse)

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


#mse=wrapper_feature_selection(selected_column)

start_time=time.clock()
selected_column=[0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
opt_loss_tracer=[]
es=cma.CMAEvolutionStrategy(selected_column, 1)
res=es.optimize(wrapper_feature_selection,iterations=1000).result
time_cost=start_time-time.clock()



'''
#filter method
'''
#录出完整components矩阵
selected_column=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','aa','ab','ad','ae','af','ag','ah','ai','aj','ak']
en_dic=be.get_the_encoding_dictionary(selected_column,e_bike_comp)
uncode_feature_vectors,feature_vectors,labels,bike_names,en_dic=be.generate_features_and_labels(labels_source,selected_column,e_bike_comp)

#转换矩阵格式,filters_mat每行第一个数组是为了保留原始编号
filters_mat=[]
for i in range(len(feature_vectors[0])):
    filters_mat.append([[i],[]])
    for j in range(len(feature_vectors)):
        filters_mat[i][1].append(feature_vectors[j][i])
filters_labels=[]   
for i in range(len(labels)):
    filters_labels.append(labels[i][0])
#计算相关度系数
for i in range(len(filters_mat)):
    coe=np.corrcoef(filters_mat[i][1],filters_labels)[0,1]
    filters_mat[i].append(coe)
    print(str(i)+' ,coe = '+str(coe))
#上面计算的是one hot的每个元素与labels的相关度，现在要聚集到filters_attribute中
filters_attribute=[]
indexer=0
for i in range(len(en_dic)):
    #跳过售价一栏，因为最后有
    if i == len(en_dic)-1:
        filters_attribute.append(['ac',[]])
    if i != len(en_dic)-1:
        filters_attribute.append([selected_column[i],[]])
    avg_coe=0
    for j in range(len(en_dic[i])):
        if np.isnan(filters_mat[indexer][2]) == True:
            print(filters_mat[indexer][2])
            avg_coe+=0
        if np.isnan(filters_mat[indexer][2]) != True:
            avg_coe+=abs(filters_mat[indexer][2])
        indexer+=1
    avg_coe=avg_coe/len(en_dic[i])
    filters_attribute[i][1].append(avg_coe)
#按相关度进行排序
filters_attribute=sorted(filters_attribute, key=lambda x:x[1],reverse = True)
#逐个增加attribute看mse的变化
new_selected_column=[]
filter_mse=[]
for i in range(len(filters_attribute)):
    new_selected_column.append(filters_attribute[i][0])
    uncode_feature_vectors,new_feature_vectors,new_labels,bike_names,new_en_dic=be.generate_features_and_labels(labels_source,new_selected_column,e_bike_comp)
    mse=k_fold_cross_validation_T_times(5,new_feature_vectors,new_labels,test_performance,5,40)
    filter_mse.append([new_selected_column,mse])
    print("attribute amount: "+str(i+1)+', mse: '+str(mse))


#filter method的plot柱状图和折线图，柱状图是相关度，折线是逐个增加时mse
name_list=[]
num_list=[]
mse_curve=[]
for i in range(len(filter_mse)):
    num_list.append(filters_attribute[i][1][0])
    name_list.append(filters_attribute[i][0])
    mse_curve.append(filter_mse[i][1])
plt.figure()
plt.bar(range(len(num_list)), num_list,tick_label=name_list)
plt.xlabel('feature names')
plt.ylabel('correlation coefficient')
plt.title('the correlation coefficient between each feature set and the label set')
plt.show()

plt.figure()
plt.plot(range(len(mse_curve)),mse_curve,label=name_list)
plt.xticks(range(len(mse_curve)),name_list) 
plt.xlabel('feature names')
plt.ylabel('MSE')
plt.title('the MSE trend when iteratively add the filter method sorted features')
plt.show()


#embedded method
regr = RandomForestRegressor(n_estimators=40,oob_score=False,bootstrap=True)
regr.fit(feature_vectors, labels)
feature_importances=regr.feature_importances_

em_attribute=[]
indexer=0
for i in range(len(en_dic)):
    if i == len(en_dic)-1:
        em_attribute.append(['ac',[feature_importances[indexer]]])
    if i != len(en_dic)-1:
        em_attribute.append([selected_column[i],[feature_importances[indexer]]])
    indexer+=1
    
em_attribute=sorted(em_attribute, key=lambda x:x[1],reverse = True)
new_selected_column=[]
em_mse=[]
for i in range(len(em_attribute)):
    new_selected_column.append(em_attribute[i][0][0])
    uncode_feature_vectors,new_feature_vectors,new_labels,bike_names,new_en_dic=be.generate_features_and_labels(labels_source,new_selected_column,e_bike_comp)
    mse=k_fold_cross_validation_T_times(5,new_feature_vectors,new_labels,test_performance,5,40)
    em_mse.append([new_selected_column,mse])
    print("attribute amount: "+str(i+1)+', mse: '+str(mse))

em_names=[]
em_importance=[]
em_mse_log=[]

for i in range(len(em_attribute)):
    em_names.append(em_attribute[i][0])
    em_importance.append(em_attribute[i][1][0])
    em_mse_log.append(em_mse[i][1])

plt.figure()
plt.bar(range(len(em_importance)), em_importance,tick_label=em_names)
plt.xlabel('feature names')
plt.ylabel('feature importance')
plt.title('the feature importance ranking of RF')
plt.show()

plt.figure()
plt.plot(range(len(em_mse_log)),em_mse_log)
plt.xticks(range(len(em_mse_log)),em_names)
plt.xlabel('feature names')
plt.ylabel('MSE')
plt.title('the MSE trend when iteratively add the embedded method sorted features')
plt.show()















