import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
from sklearn.preprocessing import StandardScaler
import tf_test
import tensorflow as tf
import random
from sklearn.metrics import mean_squared_error
random.seed(16)


def load_elektrischefietsen_data():
    bike_date_mat=[]
    file_index=1
    while(1):
        file_name='H:\\Desktop\\bike optimization model\\elektrischefietsen GA with date\\'+str(file_index)+'.csv'
        try:
            current_file= pd.read_csv(file_name,sep=',', skiprows=6)
            current_file=current_file.values
            current_file=current_file.tolist()
            #print('loaded '+str(file_index)+'.csv')
        except:
            #print('no more files')
            break
        for i in range(len(current_file)):
            bike_date_mat.append(current_file[i])
        file_index+=1
    for i in range(len(bike_date_mat)):
        try:
            bike_date_mat[i][1]=int(bike_date_mat[i][1])
            bike_date_mat[i][2]=int(re.sub(r',','',bike_date_mat[i][2]))
            bike_date_mat[i][3]=int(re.sub(r',','',bike_date_mat[i][3]))
            bike_date_mat[i][5]=float(re.sub(r'%','',bike_date_mat[i][5]))/100
            bike_date_mat[i][6]=float(re.sub(r'%','',bike_date_mat[i][6]))/100
        except:
            continue
    #print('successfully load the entire elektrischefietsen time serie data')
    time.sleep(2)
    return bike_date_mat
        

def load_fietsenwinkel_data():
    e_bikes_click_fact = pd.read_csv('H:\\Desktop\\bike optimization model\\fietsenwinkel GA with date\\M1 Content Drilldown.csv', sep=';', skiprows=1)
    e_bikes_click_fact=e_bikes_click_fact.values
    e_bikes_click_fact=e_bikes_click_fact.tolist()
    bike_date_mat = pd.read_csv('H:\\Desktop\\bike optimization model\\fietsenwinkel GA with date\\M2 Content Drilldown.csv', sep=';', skiprows=1)
    bike_date_mat=bike_date_mat.values
    bike_date_mat=bike_date_mat.tolist()
    
    for i in range(len(e_bikes_click_fact)):
        bike_date_mat.append(e_bikes_click_fact[i])
    #print('successfully load the entire fietsenwinkel time serie data')
    time.sleep(2)
    return bike_date_mat

  
def clean_data_source(bike_date_mat):
    #如果是fietsenwinkel的文件,去除不是日期的第二列
    while(1):
        del_counter=0
        for i in range(len(bike_date_mat)):
            try:
                string_checker=bike_date_mat[i][1]+'string_checker'
                string_checker=string_checker
                del bike_date_mat[i][1]
                del_counter+=1
            except:
                continue
        if del_counter == 0:
            break
    #去除是乱码的第二列,并转换第二列的格式
    new_mat=[]
    for i in range(len(bike_date_mat)):
        try:
            if np.isnan(bike_date_mat[i][3]) == True:
                continue
            try:
                if bike_date_mat[i][1] > 20180000 :
                    bike_date_mat[i][1]=str(int(bike_date_mat[i][1]))
                    new_mat.append(bike_date_mat[i])
            except:
                continue
        except:
                continue
    bike_date_mat=new_mat
    bike_date_mat=sorted(bike_date_mat, key=lambda x:x[1],reverse = False)
    #print('cleaned the entire fietsenwinkel data')
    time.sleep(2)
    return bike_date_mat


#选取我要分析的时间段
def pick_the_time_serie_for_analysis(start_date,end_date,bike_date_mat):
    time_serie_data=[]
    for i in range(len(bike_date_mat)):
        if start_date > str(int(bike_date_mat[i][1])):
            continue
        if end_date <  str(int(bike_date_mat[i][1])):
            break
        time_serie_data.append(bike_date_mat[i])
    #print('collected the time segment data from '+str(start_date)+' to '+str(end_date))
    time.sleep(2)
    return time_serie_data


#将日期转化为到end date的天数
def change_date_to_days(date,end_date):
    if int(date)%20190000 > 20180000:
        date_year=2018
        date_date=(int(date)%20180000)%100
        date_month=int(((int(date)%20180000)-(int(date)%20180000)%100)/100)
    else:
        date_year=2019
        date_date=(int(date)%20190000)%100
        date_month=int(((int(date)%20190000)-(int(date)%20190000)%100)/100)
    if int(end_date)%20190000 > 20180000:
        end_date_year=2018
        end_date_date=(int(end_date)%20180000)%100
        end_date_month=int(((int(end_date)%20180000)-(int(end_date)%20180000)%100)/100)
    else:
        end_date_year=2019
        end_date_date=(int(end_date)%20190000)%100
        end_date_month=int(((int(end_date)%20190000)-(int(end_date)%20190000)%100)/100)
    date=datetime.datetime(int(date_year),int(date_month),int(date_date))
    new_end_date=datetime.datetime(end_date_year,end_date_month,end_date_date)
    days=(new_end_date-date).days
    return days

#先检测出所有单独的自行车,排除其他无用信息
def detect_unique_bikes_logs_in_all_data(bike_date_mat):
    unique_bikes=[]
    for i in range(len(bike_date_mat)):
        bike_date_mat[i][0]=re.sub(r'[/]','',bike_date_mat[i][0])
        if '?' not in bike_date_mat[i][0] and bike_date_mat[i][0] != '':
            unique_bikes.append(bike_date_mat[i][0])
    unique_bikes=list(set(unique_bikes))
    #print('detected out all the exist unique ebikes during the time segment')
    time.sleep(2)
    return unique_bikes,bike_date_mat


#检测出每辆自行车上线的最早时间,并计算为有效天数
def detect_earliest_date_for_each_unique_bike(unique_bikes,bike_date_mat,end_date):
    for i in range(len(unique_bikes)):
        unique_bikes[i]=[unique_bikes[i]]
        #print('detect_earliest_date_for_each_unique_bike: '+str(i))
        for j in range(len(bike_date_mat)):
            if unique_bikes[i][0] == bike_date_mat[j][0]:
                days=change_date_to_days(bike_date_mat[j][1],end_date)
                if days == 0:
                    days=1
                unique_bikes[i].append(days)
                break
    unique_bikes=sorted(unique_bikes, key=lambda x:x[1],reverse = False)
    return unique_bikes


#加和每辆自行车所有点击量, 乘上了有效比例的
def sum_entire_valid_clicks(unique_bikes,bike_date_mat):
    for i in range(len(unique_bikes)):
        unique_bikes[i].append(0)
        #print('sum_entire_valid_clicks: '+str(i))
        for j in range(len(bike_date_mat)):
            if unique_bikes[i][0] == bike_date_mat[j][0]:
                try:
                    unique_bikes[i][2]+=int(int(bike_date_mat[j][3]*(1-float(bike_date_mat[j][5]))))
                except:
                    unique_bikes[i][2]+=int(int(bike_date_mat[j][3]*0.5))
                    #print('no')
    return unique_bikes

#计算labels_source               
def generate_labels_source(unique_bikes):
    labels_source=[]
    for i in range(len(unique_bikes)):
        labels_source.append([unique_bikes[i][0],unique_bikes[i][2]/unique_bikes[i][1]])
    labels_source=sorted(labels_source, key=lambda x:x[1],reverse = True)  
    #print('now you have the label source for supervised learning')
    time.sleep(2)
    return labels_source


#计算标签的主函数
def calculate_labels_source(start_date,end_date,web_source):
    if web_source == 'fietsenwinkel':
        #print('you are getting data source of www.fietsenwinkel.nl')
        #print('date: '+str(start_date)+' to '+str(end_date))
        time.sleep(2)
        bike_date_mat=load_fietsenwinkel_data()
    else:
        if web_source == 'elektrischefietsen':
            #print('you are getting data source of www.elektrischefietsen.com')
            #print('date: '+str(start_date)+' to '+str(end_date))
            time.sleep(2)
            bike_date_mat=load_elektrischefietsen_data()
    bike_date_mat=clean_data_source(bike_date_mat)
    bike_date_mat=pick_the_time_serie_for_analysis(start_date,end_date,bike_date_mat)
    unique_bikes,bike_date_mat=detect_unique_bikes_logs_in_all_data(bike_date_mat)
    unique_bikes=detect_earliest_date_for_each_unique_bike(unique_bikes,bike_date_mat,end_date)
    unique_bikes=sum_entire_valid_clicks(unique_bikes,bike_date_mat)
    labels_source=generate_labels_source(unique_bikes)
    return labels_source,unique_bikes
###############################################################################
#这一片函数是计算feature的
def alpha2num(column):
    alpha=[['a'],['b'],['c'],['d'],['e'],['f'],['g'],['h'],['i'],['j'],['k'],['l'],['m'],['n'],['o'],['p'],['q'],['r'],['s'],['t'],['u'],['v'],['w'],['x'],['y'],['z'],['aa'],['ab'],['ac'],['ad'],['ae'],['af'],['ag'],['ah'],['ai'],['aj'],['ak']]   
    for i in range(len(alpha)):
        if column == alpha[i][0]:
            return i
        
def num2alpha(num):
    alpha=[['a'],['b'],['c'],['d'],['e'],['f'],['g'],['h'],['i'],['j'],['k'],['l'],['m'],['n'],['o'],['p'],['q'],['r'],['s'],['t'],['u'],['v'],['w'],['x'],['y'],['z'],['aa'],['ab'],['ac'],['ad'],['ae'],['af'],['ag'],['ah'],['ai'],['aj'],['ak']]   
    return alpha[num][0]

def unique_elements_of_each_column(column,e_bike_comp):
    all_element=[]
    for i in range(len(e_bike_comp)):
        all_element.append(e_bike_comp[i][alpha2num(column)])  
    unique_element=list(set(all_element))
    return unique_element

def component_encoder(row,column,e_bike_comp):
    unique_element=unique_elements_of_each_column(column,e_bike_comp)
    for i in range(len(unique_element)):
        if e_bike_comp[row][alpha2num(column)] == unique_element[i]:
            return i+1
        
def get_the_encoding_dictionary(selected_column,e_bike_comp):
    en_dic=[]
    for i in range(len(selected_column)):
        en_dic.append([])
        unique_element=unique_elements_of_each_column(selected_column[i],e_bike_comp)
        for j in range(len(unique_element)):
            en_dic[i].append([unique_element[j],j])
    return en_dic
    
def create_feature_vectors(selected_column,e_bike_comp):
    feature_vectors=[]
    for i in range(len(e_bike_comp)):
        fea_vec=[]
        for j in range(len(selected_column)):
            fea_vec.append(component_encoder(i,selected_column[j],e_bike_comp))
        fea_vec.append(e_bike_comp[i][alpha2num('ac')])
        feature_vectors.append(fea_vec)
    return feature_vectors

def create_labels(labels_source,e_bike_comp):
    labels=[]
    for i in range(len(e_bike_comp)):
        labels.append(0)
        for j in range(len(labels_source)):
            if e_bike_comp[i][3] in '/'+str(labels_source[j][0])+'/'  :
                labels[i]+=labels_source[j][1]
                break
    return labels

def price_encoder(feature_vectors):
    for i in range(len(feature_vectors)):
        price_segment=100
        feature_vectors[i][-1]=round(feature_vectors[i][-1]/price_segment)
    return feature_vectors

#稀疏矩阵编码器
def sparse_matrix_encoder(feature_vectors,selected_column,e_bike_comp):
    #先检测出有多少列，和每列有多少类
    en_dic=[]
    for i in range(len(feature_vectors[0])):
        all_class=[]
        en_dic.append([])
        for j in range(len(feature_vectors)):
            all_class.append(feature_vectors[j][i])
        all_class=list(set(all_class))
        if i != len(feature_vectors[0])-1:
            comp_names=unique_elements_of_each_column(selected_column[i],e_bike_comp)
        else:
            comp_names=all_class
        for j in range(len(all_class)):
            en_dic[i].append([comp_names[j],[]])
            for z in range(len(all_class)):
                if all_class[z] != all_class[j]:
                    en_dic[i][j][1].append(0)
                else:
                    en_dic[i][j][1].append(1)
        for j in range(len(feature_vectors)):
            encoder=[]
            for z in range(len(all_class)):
                if all_class[z] != feature_vectors[j][i]:
                    encoder.append(0)
                else:
                    encoder.append(1)
            
            feature_vectors[j][i]=encoder
    for i in range(len(feature_vectors)):
        combinator=[]
        for j in range(len(feature_vectors[i])):
            combinator+=feature_vectors[i][j]
        feature_vectors[i]=combinator
    
    return feature_vectors,en_dic

def generate_features_and_labels(labels_source,selected_column,e_bike_comp):
    original_feature_vectors=create_feature_vectors(selected_column,e_bike_comp)
    uncode_feature_vectors=price_encoder(original_feature_vectors)
    feature_vectors,en_dic=sparse_matrix_encoder(uncode_feature_vectors,selected_column,e_bike_comp)
    '''
    std = StandardScaler()
    std.fit(uncode_feature_vectors)
    feature_vectors=std.fit_transform(uncode_feature_vectors)
    '''
    #测试数据加入全部feature集一起编码，然后提出来
    labels=create_labels(labels_source,e_bike_comp)
    bike_names=[]
    new_fea=[]
    new_la=[]
    for i in range(len(labels)):
        if labels[i] != 0:
            bike_names.append(e_bike_comp[i][3])
            new_fea.append(feature_vectors[i])
            new_la.append([labels[i]])
    x_data=np.array(new_fea)
    y_data=np.zeros((len(new_la),1))
    
    for i in range(len(new_la)):
        #归一化用这个
        #y_data[i][0]=new_la[i][0]/max(new_la)[0]
        #不归一化用这个
        y_data[i][0]=new_la[i][0]/max(new_la)[0]
    #print(max(new_la)[0])
    
    return uncode_feature_vectors,x_data,y_data,bike_names,en_dic

def generate_dummy_features_and_labels(labels_source,selected_column,e_bike_comp):
    original_feature_vectors=create_feature_vectors(selected_column,e_bike_comp)
    uncode_feature_vectors=price_encoder(original_feature_vectors)
    #feature_vectors,en_dic=sparse_matrix_encoder(uncode_feature_vectors,selected_column,e_bike_comp)
    
    std = StandardScaler()
    std.fit(uncode_feature_vectors)
    feature_vectors=std.fit_transform(uncode_feature_vectors)
    
    #测试数据加入全部feature集一起编码，然后提出来
    labels=create_labels(labels_source,e_bike_comp)
    bike_names=[]
    new_fea=[]
    new_la=[]
    for i in range(len(labels)):
        if labels[i] != 0:
            bike_names.append(e_bike_comp[i][3])
            new_fea.append(feature_vectors[i])
            new_la.append([labels[i]])
    x_data=np.array(new_fea)
    y_data=np.zeros((len(new_la),1))
    
    for i in range(len(new_la)):
        y_data[i][0]=new_la[i][0]/max(new_la)[0]
    #print(max(new_la)[0])
    en_dic=[]
    return uncode_feature_vectors,x_data,y_data,bike_names,en_dic

def encode_test_data(uncode_feature_vectors,test_data):
    price_segment=100
    for i in range(len(test_data)):
        test_data[i][-1]=round(test_data[i][-1]/price_segment)
    for i in range(len(test_data)):
        uncode_feature_vectors.append(test_data[i])
    std = StandardScaler()
    std.fit(uncode_feature_vectors)
    encoded_feature_vectors=std.fit_transform(uncode_feature_vectors)
    encoded_test_data=[]
    for i in range(len(test_data)):
        encoded_test_data.append(encoded_feature_vectors[len(encoded_feature_vectors)-(i+1)])
    encoded_test_data=np.array(encoded_test_data)
    del uncode_feature_vectors[len(uncode_feature_vectors)-len(encoded_test_data)]
    return encoded_test_data

#交叉验证，随机分成k份，可重复,一份作为测试集
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

def plot_test_set_result(test_y,test_pred):
    plt.figure()
    x_plot=[]
    original_y=[]
    predict_y=[]
    compare=[]
    for i in range(len(test_y)):
        x_plot.append(i)
        compare.append([test_y[i][0],test_pred[i][0]])
    compare=sorted(compare, key=lambda x:x[0],reverse = True)
    for i in range(len(compare)):
        original_y.append(compare[i][0])
        predict_y.append(compare[i][1])
    plt.plot(x_plot,original_y,color='b',label = 'real score')    
    plt.plot(x_plot,predict_y,'g-',label = 'predicted score')
    plt.title('The trend relationship between the real score and the predicted score')     #标题
    plt.xlabel('bikes')       #x轴的标签
    plt.ylabel('average valid clicks per day')       #y轴的标签
    plt.legend()            #设置图例
    plt.show()
    
    plt.figure()
    plt.plot(original_y,original_y,'b-',label = 'real score')
    plt.scatter(original_y,predict_y,marker='x',color='g',label = 'predict score',alpha=0.6)
    plt.title('The relationship between the real_score and the predict_score')
    plt.xlabel('y labels')       #x轴的标签
    plt.ylabel('y predict')       #y轴的标签
    plt.legend()            #设置图例
    plt.show()
    
def evaluate_e_bike(uncode_feature_vectors,evaluation_data):
    evaluation_data=encode_test_data(uncode_feature_vectors,evaluation_data)
    bike_evaluation_result=tf_test.predict_with_the_trained_model(evaluation_data)
    #print('bike scored: '+str(bike_evaluation_result))
    return bike_evaluation_result[0][0]
    
#creating time series data
'''import label_sources.spydata'''
def creating_time_series_data(start_date,end_date,source):
    if source=='elektrischefietsen':
        elek_labels_source,elek_unique_bikes=calculate_labels_source(start_date,end_date,source)
        labels_source=elek_labels_source
    else:
        fiets_labels_source,fiets_unique_bikes=calculate_labels_source(start_date,end_date,'fietsenwinkel')
        labels_source=fiets_labels_source 
    e_bike_comp = pd.read_csv('e_bike_excel_copy.csv',sep=';')
    e_bike_comp=e_bike_comp.values
    e_bike_comp=e_bike_comp.tolist()
    selected_column=['b','c','g','aa','t','j','f','x','e','r','ak']
    #selected_column=[b:brand, c:frame-color, g:wheel-color, aa:frame-type, t:battery-position, j:engine-position, f:front-carrier, x:bike-type, e:saddle-and-bar-color, and price]
    en_dic=get_the_encoding_dictionary(selected_column,e_bike_comp)
    #uncode_feature_vectors,feature_vectors,labels,bike_names,en_dic=generate_dummy_features_and_labels(labels_source,selected_column,e_bike_comp)
    uncode_feature_vectors,feature_vectors,labels,bike_names,en_dic=generate_features_and_labels(labels_source,selected_column,e_bike_comp)
    return selected_column,labels_source,en_dic,uncode_feature_vectors,feature_vectors,labels,bike_names


#train the model(if you didn't change the data source then you don't have to run this function)
def train_the_model(feature_vectors,labels,bike_names,k=10,num_for_each_k=350,num_nodes_L1=120,num_nodes_L2=30,num_train=1000,step_size=0.2):
    num_features=len(feature_vectors[0])
    #900 is the best
    final_loss_set=[]
    train_x_set,train_y_set,test_x,test_y=creat_random_cross_valida_sets(feature_vectors,labels,k,num_for_each_k)
    tf.reset_default_graph()
    compare,colorful_list,loss_tracer=tf_test.Neural_Networks_Nonlinear_Regression(train_x_set,train_y_set,test_x,test_y,bike_names,num_features,num_nodes_L1,num_nodes_L2,num_train,step_size)
    test_pred=tf_test.predict_with_the_trained_model(feature_vectors)
    plot_test_set_result(test_y,test_pred)
    #print('after '+str(k)+' cross validation trained, the loss reached: '+str(loss_tracer[-1]))
    final_loss_set.append(loss_tracer[-1])
    return final_loss_set
    
#you just need to run this row if you need different time serie data source
def update_the_model_with_different_time_series_data(start_date,end_date,source):
    selected_column,labels_source,en_dic,uncode_feature_vectors,feature_vectors,labels,bike_names=creating_time_series_data(start_date,end_date,source)
    final_loss_set=train_the_model(feature_vectors,labels,bike_names)
    return selected_column,labels_source,en_dic,uncode_feature_vectors,feature_vectors,labels,bike_names,final_loss_set


def train_ANN(k=10,num_for_each_k=350,num_nodes_L1=120,num_nodes_L2=30):
    step_size=0.2
    num_train=1000
    num_features=len(feature_vectors[0])
    #900 is the best
    train_x_set,train_y_set,test_x,test_y=creat_random_cross_valida_sets(feature_vectors,labels,k,num_for_each_k)
    if test_x == []:
        num_for_each_k=350
        k=50
    tf.reset_default_graph()
    compare,colorful_list,loss_tracer=tf_test.Neural_Networks_Nonlinear_Regression(train_x_set,train_y_set,test_x,test_y,bike_names,num_features,num_nodes_L1,num_nodes_L2,num_train,step_size)
    test_pred=tf_test.predict_with_the_trained_model(test_x)
    print(mean_squared_error(test_y, test_pred))
    opt_loss_tracer.append(mean_squared_error(test_y, test_pred))
    return mean_squared_error(test_y, test_pred)

#k倍交叉验证，适用于所有类型的训练function(train_x,train_y,test_x,test_y),function的输出为mse
def k_fold_cross_validation_T_times(k,X_set,Y_set,function,T):
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
            mse_set.append(function(train_x,train_y,test_x,test_y))
        avg_mse=np.mean(mse_set)
        t_times_mse.append(avg_mse)
    t_times_avg_mse=np.mean(t_times_mse)
    return t_times_avg_mse
    











