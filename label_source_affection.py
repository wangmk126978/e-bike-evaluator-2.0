
import datetime
import time
import re
import kangkang_tools_box as kk
import bike_evaluator as be
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import numpy as np
import random

def date_2_string(date):
    string_list=kk.check_symbol('-',re.sub(r' 00:00:00','',str(date)))
    string=string_list[0]+string_list[1]+string_list[2]
    return string

def time_duration_perlong(start_date,end_date):
    time_spot_list=[date_2_string(start_date)]
    time_spot=start_date
    while(time_spot+datetime.timedelta(days=30)<end_date):
        time_spot+=datetime.timedelta(days=30)
        time_spot_list.append(date_2_string(time_spot))
    return time_spot_list

def get_mse(feature_vectors,labels):
    regr = RandomForestRegressor(n_estimators=200,oob_score=True)
    regr.fit(feature_vectors, labels)  
    oob=1 - regr.oob_score_ 
    return oob

def string_2_date(date):
    if int(date)%20190000 > 20180000:
        date_year=2018
        date_date=(int(date)%20180000)%100
        date_month=int(((int(date)%20180000)-(int(date)%20180000)%100)/100)
    else:
        date_year=2019
        date_date=(int(date)%20190000)%100
        date_month=int(((int(date)%20190000)-(int(date)%20190000)%100)/100)
    date=datetime.datetime(int(date_year),int(date_month),int(date_date))
    return date

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

    



source='elektrischefietsen'
start_date=datetime.datetime(int('2018'),int('10'),int('1'))
end_date=datetime.datetime(int('2019'),int('7'),int('10'))
days_delta=datetime.timedelta(days=1)
time_spot_list=time_duration_perlong(start_date,end_date)

#不同time span和不同时间点MSE的变化
duration_shift=[]
duration=30
index=0
while(duration+30 < (end_date-start_date).days):
    duration_shift.append([])
    for i in range(len(time_spot_list)):
        if string_2_date(time_spot_list[i])+datetime.timedelta(days=duration) < end_date:
            selected_column,labels_source,en_dic,uncode_feature_vectors,feature_vectors,labels,bike_names=be.creating_time_series_data(time_spot_list[i],date_2_string(string_2_date(time_spot_list[i])+datetime.timedelta(days=duration)),source)
            print('duration: '+str(duration)+', start date: '+str(time_spot_list[i]))
            mse=k_fold_cross_validation_T_times(5,feature_vectors,labels,test_performance,5,40)
            duration_shift[index].append([duration+30*(1+i),mse])
            print("time_span: "+str(duration)+", start_date: "+str(time_spot_list[i])+", mse: "+str(mse))
        else:
            break
    duration+=30
    index+=1
for i in range(len(duration_shift)):
    duration=0
    for j in range(len(duration_shift[i])):
        duration_shift[i][j][0]=duration
        duration+=30
duration_shift.append([])
selected_column,labels_source,en_dic,uncode_feature_vectors,feature_vectors,labels,bike_names=be.creating_time_series_data(date_2_string(start_date),date_2_string(end_date),source)
duration_shift[len(duration_shift)-1].append([0,k_fold_cross_validation_T_times(5,feature_vectors,labels,test_performance,5,40)])

avg_X=[]
avg_Y=[]
plt.figure()
for i in range(len(duration_shift)):
    X=[]
    Y=[]
    label='time span(s) = '+str(30*(i+1))
    avg_X.append(30*(i+1))
    for j in range(len(duration_shift[i])):
        X.append(duration_shift[i][j][0])
        Y.append(duration_shift[i][j][1])
    avg_Y.append(np.mean(Y))
    plt.scatter(X,Y,label = label,marker='v')
ave_MSE_d=[]
d=[]
for i in range(len(duration_shift[0])):
    sum_one_s=0
    num_s=0
    for j in range(len(duration_shift)):
        try:
            sum_one_s+=duration_shift[j][i][1]
            num_s+=1
        except:
            break
    ave_MSE_d.append(sum_one_s/num_s)
    d.append(30*i)
start_date_coef=np.corrcoef([ave_MSE_d,d])
plt.plot(d,ave_MSE_d,label='mean(MSE_st)')
#plt.title('The relationship between the duration settings and the MSE of RF(n=200)')     #标题
plt.xlabel('start date(d)')       #x轴的标签
plt.ylabel('MSE')       #y轴的标签
plt.legend()            #设置图例
plt.show()

plt.figure()
plt.plot(avg_X,avg_Y)
plt.title('The relationship between time span(s) and the average MSE of RF(n=200)')
plt.xlabel('time span(s)')       #x轴的标签
plt.ylabel('average MSE of RF') 
plt.show()


#计算startdate和MSE之间的关系


plt.figure()
plt.plot(d,d)
plt.plot(d,ave_MSE_d)
plt.show()



#计算number of trees的影响
#节约时间只算第一个时间点的不同time span的200的曲线

time_span=30
MSE_ts=[]
index=0
while(time_span+30 < (end_date-start_date).days):
    MSE_ts.append([['time span=',time_span],[]])
    selected_column,labels_source,en_dic,uncode_feature_vectors,feature_vectors,labels,bike_names=be.creating_time_series_data(time_spot_list[0],date_2_string(string_2_date(time_spot_list[0])+datetime.timedelta(days=time_span)),source)
    for i in range(100):
        mse=k_fold_cross_validation_T_times(5,feature_vectors,labels,test_performance,5,i+1)
        MSE_ts[index][1].append(mse)
        print('time span: '+str(time_span)+', n = '+str(i)+', mse = '+str(mse))
    time_span+=30
    index+=1
#加上最后一个时段的
MSE_ts.append([['time span=',time_span],[]])
selected_column,labels_source,en_dic,uncode_feature_vectors,feature_vectors,labels,bike_names=be.creating_time_series_data(time_spot_list[0],date_2_string(string_2_date(time_spot_list[0])+datetime.timedelta(days=270)),source)
for i in range(100):
    mse=k_fold_cross_validation_T_times(5,feature_vectors,labels,test_performance,5,i+1)
    MSE_ts[len(MSE_ts)-1][1].append(mse)
    print('time span: '+str(time_span)+', n = '+str(i)+', mse = '+str(mse))

plt.figure()
for i in range(len(MSE_ts)):
    X=list(range(100))
    Y=MSE_ts[i][1]
    label='time span(s) = '+str(MSE_ts[i][0][1])
    plt.plot(X,Y,label=label)
    #plt.title('How the number of decision trees(n) setting affect RF MSE')
    plt.xlabel('number of decision trees(n)')       #x轴的标签
    plt.ylabel('MSE') 
    plt.legend() 
plt.show()
    




#计算n=40时各个time span的影响
X=[]
Y=[]
for i in range(len(MSE_ts)):
    X.append(MSE_ts[i][0][1])
    Y.append(MSE_ts[i][1][40])
plt.figure()
plt.plot(X,Y)
#plt.title('How the time span(s) setting affect RF MSE')
plt.xlabel('time span(s)')       #x轴的标签
plt.ylabel('MSE') 
plt.legend() 











'''
#时间段延长
time_spot_list=time_duration_perlong(start_date,end_date)

extend_mse_tracer=[]
for i in range(len(time_spot_list)+1):
    try:
        selected_column,labels_source,en_dic,uncode_feature_vectors,feature_vectors,labels,bike_names=be.creating_time_series_data(date_2_string(start_date),time_spot_list[i],source)
    except:
        selected_column,labels_source,en_dic,uncode_feature_vectors,feature_vectors,labels,bike_names=be.creating_time_series_data(date_2_string(start_date),date_2_string(end_date),source)
    extend_mse_tracer.append(get_mse(feature_vectors,labels))
    
shift_mse_tracer=[]
for i in range(len(time_spot_list)+1):
    try:
        selected_column,labels_source,en_dic,uncode_feature_vectors,feature_vectors,labels,bike_names=be.creating_time_series_data(date_2_string(start_date),time_spot_list[i],source)
    except:
        selected_column,labels_source,en_dic,uncode_feature_vectors,feature_vectors,labels,bike_names=be.creating_time_series_data(date_2_string(start_date),date_2_string(end_date),source)
    shift_mse_tracer.append(get_mse(feature_vectors,labels))
    
duration=[]
for i in range(len(time_spot_list)+1):
    duration.append(30*(1+i))
plt.plot(duration,mse_tracer,'b',label = 'original y')
plt.title('Time period length(days) affection on MSE of random forest OOBs')
plt.xlabel('Time period length(days)')       #x轴的标签
plt.ylabel('MSE of random forest OOBs')  
'''