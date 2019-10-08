import bike_evaluator as be

#selected_column,labels_source,en_dic,uncode_feature_vectors,feature_vectors,labels,bike_names,final_loss_set=update_the_model_with_different_time_series_data(start_date='20181001',end_date='20190710',source='elektrischefietsen')

start_date='20181001'
end_date='20190710'
source='elektrischefietsen'
selected_column,labels_source,en_dic,uncode_feature_vectors,feature_vectors,labels,bike_names=be.creating_time_series_data(start_date,end_date,source)

final_loss_set=be.train_the_model(feature_vectors,labels,bike_names,k=900,num_for_each_k=350,num_nodes_L1=120,num_nodes_L2=30,num_train=1000,step_size=0.2)

###############################################################################
'''
尝试一下标准的boosting 方法与我们原来的方法的比较
我们的方法是变相的袋装法，只是所有的袋子都用来循环地训练同一个模型
boosting是保证训练集不变，重复地训练900次
'''
#保持x_training_set有900个相同的矩阵
from sklearn.metrics import mean_squared_error
import tf_test

boost_x_set=[]
boost_y_set=[]
for i in range(200):
    boost_x_set.append(removed_train_x)
    boost_y_set.append(removed_train_y)
boost_x_set=np.array(boost_x_set)
boost_y_set=np.array(boost_y_set)

tf.reset_default_graph()
compare,colorful_list,loss_tracer=tf_test.Neural_Networks_Nonlinear_Regression(boost_x_set,boost_y_set,test_x,test_y,bike_names,len(feature_vectors[0]),120,30,1000,0.2)
test_pred=tf_test.predict_with_the_trained_model(test_x)
be.plot_test_set_result(test_y,test_pred)
print(mean_squared_error(test_y,test_pred))


###############################################################################
'''
#run this 2 rows to evaluate an ebike
evaluation_data=[[0,0,0,0,0,0,0,0,1,1,2500]]
#need to fill in [[b:brand, c:frame-color, g:wheel-color, aa:frame-type, t:battery-position, j:engine-position, f:front-carrier, x:bike-type, e:saddle-and-bar-color , price]]
#refer to the en_dic
evaluate_e_bike(uncode_feature_vectors,evaluation_data)
'''

#将brinkers和Victesse选出来分析并排序
#要加上[url,预测值,真实值,真实销量，真实网站点击率]

IBG_e_bikes=[]
max_labels_source=[]
for i in range(len(bike_names)):
    print(i)
    if 'brinckers' in  bike_names[i] or 'victesse' in bike_names[i]:
        for j in range(len(hyperlink_mat)):
            if hyperlink_mat[j][1] == bike_names[i]:
                IBG_e_bikes.append([bike_names[i],round(tf_test.predict_with_the_trained_model([feature_vectors[i]])[0][0],3),round(labels[i][0],3),hyperlink_mat[j][2][0],hyperlink_mat[j][2][1]])
                break
            if j == len(hyperlink_mat)-1:
                IBG_e_bikes.append([bike_names[i],round(tf_test.predict_with_the_trained_model([feature_vectors[i]])[0][0],3),round(labels[i][0],3),0,0])

IBG_e_bikes=sorted(IBG_e_bikes, key=lambda x:x[1],reverse = True)




test_result=[]
for i in range(len(feature_vectors)):  
    print(tf_test.predict_with_the_trained_model([feature_vectors[i]])[0],labels[i][0])
    test_result.append([tf_test.predict_with_the_trained_model([feature_vectors[i]])[0],labels[i][0]])
    
#查看display oreder相关度
order=[]
y=[]
for i in range(len(labels)):
    y.append(labels[i][0])
    order.append(i)
plt.plot(order,y,color='black',label = 'real score')
plt.title('The relationship between the display order and the popularity score')
plt.xlabel('display order')       #x轴的标签
plt.ylabel('popularity score') 
a=np.corrcoef([order,y])