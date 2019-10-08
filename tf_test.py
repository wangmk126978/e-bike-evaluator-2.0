import re
import tensorflow as tf
import numpy as np
import csv
import random
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import random
from sklearn.metrics import mean_squared_error
random.seed(16)

###############################################################################
'''多元非线性回归神经网络'''


#x_data = np.zeros((200,num_features))
#for i in range(len(x_data)):
#    for j in range(len(x_data[i])):
#        x_data[i][j]=random.uniform(-1,1)
#noise = np.random.normal(0,0.02,x_data.shape)
#y_data=np.zeros((x_data.shape[0],1))
#for i in range(len(y_data)):
#    y_data[i][0]=i/len(y_data)
 
'''
#源代码给的数据集
x_data = np.linspace(-0.5,0.5,200)[:,np.newaxis]
noise = np.random.normal(0,0.02,x_data.shape)
y_data = np.square(x_data) + noise

#我要测试的数据集
x_data=np.array(feature_vectors)
y_data=np.zeros((len(labels),1))
for i in range(len(labels)):
    y_data[i][0]=labels[i]/max(labels)
    
x_data=np.array(new_fea)
y_data=np.zeros((len(new_la),1))
for i in range(len(new_la)):
    y_data[i][0]=labels[i]/max(new_la)
test_data=np.array(test_data)
'''



def Neural_Networks_Nonlinear_Regression(train_x_set,train_y_set,test_x,test_y,bike_names,num_features,num_nodes_L1,num_nodes_L2,num_train,step_size):
    print('now train the Neural Networks')
    #定义两个placeholder(占位符)，规定是1列
    x = tf.placeholder(tf.float32,[None,num_features],name='x')
    y = tf.placeholder(tf.float32,[None,1],name='y')
     
    #使用神经网络进行训练测试
     
    #定义神经网络的中间层（隐藏层）
    #权重
    Weights_L1 = tf.Variable(tf.random_normal([num_features,num_nodes_L1]),name='w_l1')
    #偏置
    biases_L1 = tf.Variable(tf.zeros([num_nodes_L1]),name='b_l1')
    #传入中间层（隐藏层）的值
    Wx_plus_b_L1 = tf.matmul(x, Weights_L1) + biases_L1
    #由中间层（隐藏层）输出的值,激活函数使用双曲正切函数
    #L1 = tf.nn.tanh(Wx_plus_b_L1)
    L1 = tf.nn.relu(Wx_plus_b_L1)
     
    #定义神经网络的输出层
    Weights_L2 = tf.Variable(tf.random_normal([num_nodes_L1,num_nodes_L2]),name='w_l2')
    biases_L2 = tf.Variable(tf.zeros([num_nodes_L2]),name='b_l2')
    Wx_plus_b_L2 = tf.matmul(L1,Weights_L2) + biases_L2
    #L2=tf.nn.tanh(Wx_plus_b_L2)
    L2 = tf.nn.relu(Wx_plus_b_L2)
    
    Weights_L_Out = tf.Variable(tf.random_normal([num_nodes_L2,1]),name='w_lout')
    biases_L_Out = tf.Variable(tf.zeros([1]),name='b_lout')
    Wx_plus_b_L_Out = tf.matmul(L2,Weights_L_Out) + biases_L_Out
    prediction = tf.nn.sigmoid(Wx_plus_b_L_Out)
     
    #定义二次代价函数
    loss = tf.reduce_mean(tf.square(y - prediction))
    #使用梯度下降法训练，最小话代价函数
    train_step = tf.train.GradientDescentOptimizer(step_size).minimize(loss)
     
    #初始化变量
    init = tf.global_variables_initializer()
    #保存模型的函数
    saver = tf.train.Saver()
    tf.add_to_collection("L1", L1)
    tf.add_to_collection("L2", L2)
    tf.add_to_collection("prediction", prediction)
    #定义会话
    with tf.Session() as sess:
        sess.run(init)
        loss_tracer=[]
#        best_loss=1
        current_loss=0
        all_y_train=[]
        for i in range(len(train_x_set)):
            print('folder '+str(i+1))
            x_data=train_x_set[i]
            y_data=train_y_set[i]
            for _ in range(num_train):
                #feed操作，需要参数的时候再传入
                sess.run(train_step,feed_dict = {x:x_data,y:y_data})
                current_loss=sess.run(loss,feed_dict = {x:x_data,y:y_data})
                loss_tracer.append(current_loss)
        #        if current_loss < best_loss:
        #            best_loss=current_loss
        #            loss_tracer.append(current_loss)
        #            if current_loss < 0.1:
        #                print(_)
        #                break
        
            #获得预测值
            compare=[]
            '''
            这段也是有用的
            y_pred = sess.run(prediction,feed_dict = {x:test_x})
            
            for j in range(len(test_y)):
                compare.append([test_y[j][0],y_pred[j][0]])
            compare=sorted(compare, key=lambda x:x[0],reverse = True)
            compare=np.array(compare)
            for j in range(len(compare)):
                all_y_train.append(compare[j])
            print('len oob= '+str(len(compare))+', mse: '+str(mean_squared_error(compare[:,0], compare[:,1], multioutput='raw_values')[0]))
            '''
        saver.save(sess, './tf_saver/model.ckpt')
        #绘图显示
    #    plt.figure()
    #    plt.scatter(x_data,y_data)
    #    plt.plot(x_data,predicton_value,'r-',lw = 5)
    #    plt.show()
    '''
    这段是有用的
    plt.figure()
    x_plot=[]
    for i in range(len(loss_tracer)):
        x_plot.append(i)
    plt.scatter(x_plot,loss_tracer)
    plt.show()
    
    plt.figure()
    original_y=[]
    predict_y=[]
    x_plot=[]
    for i in range(len(all_y_train)):
        x_plot.append(i)
        original_y.append(all_y_train[i][0])
        predict_y.append(all_y_train[i][1])
    plt.plot(x_plot,original_y,'b-',label = 'original y')
    plt.plot(x_plot,predict_y,'r-',label = 'predicted y')
    plt.title('The trend relationship between the original y and the predicted y')     #标题
    plt.xlabel('bikes')       #x轴的标签
    plt.ylabel('average valid clicks per day')       #y轴的标签
    plt.legend()            #设置图例
    plt.show()
    '''
    
    colorful_list=[]
    '''
    这段也是有用的
    for i in range(len(y_pred)):
        colorful_list.append([y_data[i][0],y_pred[i][0]])
    colorful_list=np.array(colorful_list)
    '''
    return compare,colorful_list,loss_tracer


def predict_with_the_trained_model(test_data):
    with tf.Session() as sess:  
        saver=tf.train.import_meta_graph('./tf_saver/model.ckpt.meta')
        saver.restore(sess,tf.train.latest_checkpoint("tf_saver/")) 
        prediction= tf.get_collection("prediction")
        graph = tf.get_default_graph()
        x = graph.get_operation_by_name("x").outputs[0]
        re_pred=sess.run(prediction,feed_dict = {x:test_data})[0]
    return re_pred
#compare,colorful_list,test_pred,loss_tracer=Neural_Networks_Nonlinear_Regression(x_data,y_data,test_data,bike_names,num_features=10,num_nodes_L1=120,num_nodes_L2=30,num_train=20000,step_size=0.2)


#'''
#看怎么保存模型
#封装目标函数
#写几个连续变化的参数集，看看运行效果
#尝试几种优化方法，证明靠谱
#'''
#
#from tensorflow.examples.tutorials.mnist import input_data
##还要写模型的保存哟！！！！！！！！！！！！！！！！！！！！
#
#INPUT_NODE = 5 # 输入节点数
#OUTPUT_NODE = 1 # 输出节点数
#LAYER1_NODE = 30 # 隐含层节点数
#BATCH_SIZE = 10
#LEARNING_RETE_BASE = 10.0 # 基学习率
#LEARNING_RETE_DECAY = 0.9 # 学习率的衰减率
#REGULARIZATION_RATE = 0.0001 # 正则化项的权重系数
#TRAINING_STEPS = 10000 # 迭代训练次数
#MOVING_AVERAGE_DECAY = 0.99 # 滑动平均的衰减系数
#train_num=3#训练集大小
# 
## 传入神经网络的权重和偏置，计算神经网络前向传播的结果
#def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
#    # 判断是否传入ExponentialMovingAverage类对象
#    if avg_class == None:
#        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
#        return tf.matmul(layer1, weights2) + biases2
#    else:
#        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1))
#                                      + avg_class.average(biases1))
#        return tf.matmul(layer1, avg_class.average(weights2))\
#                         + avg_class.average(biases2)
# 
## 神经网络模型的训练过程
#def train(input_vec,label_class):
#    x = tf.placeholder(tf.float32, [None,INPUT_NODE], name='x-input')
#    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')
# 
#    # 定义神经网络结构的参数
#    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE],
#                                               stddev=0.1))
#    biases1  = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
#    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE],
#                                               stddev=0.1))
#    biases2  = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))
# 
#    # 计算非滑动平均模型下的参数的前向传播的结果
#    y = inference(x, None, weights1, biases1, weights2, biases2)
#    
#    global_step = tf.Variable(0, trainable=False) # 定义存储当前迭代训练轮数的变量
# 
#    # 定义ExponentialMovingAverage类对象
#    variable_averages = tf.train.ExponentialMovingAverage(
#                        MOVING_AVERAGE_DECAY, global_step) # 传入当前迭代轮数参数
#    # 定义对所有可训练变量trainable_variables进行更新滑动平均值的操作op
#    variables_averages_op = variable_averages.apply(tf.trainable_variables())
# 
#    # 计算滑动模型下的参数的前向传播的结果
#    average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2)
# 
#    # 定义交叉熵损失值
#    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
#                    logits=y, labels=tf.argmax(y_, 1))
#    cross_entropy_mean = tf.reduce_mean(cross_entropy)
#    # 定义L2正则化器并对weights1和weights2正则化
#    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
#    regularization = regularizer(weights1) + regularizer(weights2)
#    loss = cross_entropy_mean + regularization # 总损失值
# 
#    # 定义指数衰减学习率
#    learning_rate = tf.train.exponential_decay(LEARNING_RETE_BASE, global_step,
#                    train_num / BATCH_SIZE, LEARNING_RETE_DECAY)
#    # 定义梯度下降操作op，global_step参数可实现自加1运算
#    train_step = tf.train.GradientDescentOptimizer(learning_rate)\
#                         .minimize(loss, global_step=global_step)
#    # 组合两个操作op
#    train_op = tf.group(train_step, variables_averages_op)
#    '''
#    # 与tf.group()等价的语句
#    with tf.control_dependencies([train_step, variables_averages_op]):
#        train_op = tf.no_op(name='train')
#    '''
#    # 定义准确率
#    # 在最终预测的时候，神经网络的输出采用的是经过滑动平均的前向传播计算结果
#    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
#    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# 
#    # 初始化回话sess并开始迭代训练
#    with tf.Session() as sess:
#        sess.run(tf.global_variables_initializer())
#        # 验证集待喂入数据
#        validate_feed = {x: input_vec, y_: label_class}
#        # 测试集待喂入数据
#        #test_feed = {x: mnist.test.images, y_: mnist.test.labels}
#        for i in range(TRAINING_STEPS):
#            if i % 1000 == 0:
#                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
#                print('After %d training steps, validation accuracy'
#                      ' using average model is %f' % (i, validate_acc))
#            xs=input_vec
#            ys=label_class
#            sess.run(train_op, feed_dict={x: xs, y_:ys})
#            
#            a=sess.run(y,feed_dict={x:xs})
#            b=[]
#            for j in range(len(ys)):
#                b.append([float(a[j][0]),float(ys[j][0])])
#            b=np.array(b)
#            print(i)
#            print(b)
#        return b
#        '''
#        test_acc = sess.run(accuracy, feed_dict=test_feed)
#        print('After %d training steps, test accuracy'
#              ' using average model is %f' % (TRAINING_STEPS, test_acc))
#        '''
#'''
## 主函数
##将词库转换成向量
#sentences = word2vec.Text8Corpus('train_library.txt')
#model = word2vec.Word2Vec(sentences,size=100)
##model.most_similar(positive=['driver'], topn=10)
#    
#    
##先将类别编号，把标签文件读进来，然后选出在词库里的词语作为训练集
#data=[]
#with open("labeled_source_big_class.csv",'r') as f1:
#    reader = csv.reader(f1)
#    for row in reader:
#        data.append(row[0])
#data.remove(data[0])
#train_set=[]
#for i in range(len(data)):
#    train_set.append([re.split(r';', data[i])[0],re.split(r';', data[i])[1]])
##创造训练集
#input_vec=[]
#label_class=[]
#intraining_words=[]
#for i in range(len(train_set)):
#    try:
#        input_vec.append(model[train_set[i][0]])
#        label_class.append(train_set[i][1])
#        intraining_words.append(train_set[i])
#    except:
#        continue
#input_vec=np.array(input_vec)
#y_data=np.zeros((len(label_class),1))
#for i in range(len(label_class)):
#    y_data[i][0]=int(label_class[i])
#label_class=y_data
#train(input_vec, label_class)
#'''
#
#X_train=np.array(feature_vectors)
#Y_train=labels
#y_data=np.zeros((len(Y_train),1))
#for i in range(len(Y_train)):
#    y_data[i][0]=Y_train[i]
#Y_train=y_data
#a=train(X_train, label_class)
#'''
##预测:
#with tf.Session() as sess:
#    sess.run(tf.global_variables_initializer())
#    #xs是需要预测的输入值
#    a=sess.run(y,feed_dict={x:input_vec})
#'''
#
#
#
#
#
################################################################################
#'''靠谱的TensorFlow梯度下降的多元回归预测'''
#import pandas as pd
#import numpy as np
#import statsmodels.formula.api as smf
#import tensorflow as tf
#import matplotlib.pyplot as plt
# 
#X_train=np.array(feature_vectors)
#Y_train=labels
#y_data=np.zeros((len(Y_train),1))
#for i in range(len(Y_train)):
#    y_data[i][0]=Y_train[i]
#Y_train=y_data
# 
#rng = np.random
# 
#learning_rate = 0.3
#training_epochs = 1000
# 
#training_data = X_train
#training_label = Y_train
#testing_data = training_data
#testing_label = training_label
# 
#n_samples = training_data.shape[0]
#X = tf.placeholder(tf.float32)
#X2 = tf.placeholder(tf.float32)
#X3 = tf.placeholder(tf.float32)
#X4 = tf.placeholder(tf.float32)
#X5 = tf.placeholder(tf.float32)
#Y = tf.placeholder(tf.float32)
# 
#w = tf.Variable(rng.randn(),name="weights",dtype=tf.float32)
#w2 = tf.Variable(rng.randn(),name="weights",dtype=tf.float32)
#w3 = tf.Variable(rng.randn(),name="weights",dtype=tf.float32)
#w4 = tf.Variable(rng.randn(),name="weights",dtype=tf.float32)
#w5 = tf.Variable(rng.randn(),name="weights",dtype=tf.float32)
#b = tf.Variable(rng.randn(),name="biases",dtype=tf.float32)
# 
#pred = tf.multiply(X,w)+tf.multiply(X2,w2)+tf.multiply(X3,w3)+tf.multiply(X4,w4)+tf.multiply(X5,w5)+b
#cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)
##优化器
#optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
# 
#init = tf.global_variables_initializer()
# 
#with tf.Session() as sess:
# 
#    sess.run(init)
# 
#    for epoch in range(training_epochs):
# 
#        sess.run(optimizer,feed_dict={X:training_data[:,0], X2:training_data[:,1], X3:training_data[:,2], X4:training_data[:,3], X5:training_data[:,4], Y:training_label})
#        if epoch%100 == 0:
#                predict_y = sess.run(pred,feed_dict={X:training_data[:,0], X2:training_data[:,1], X3:training_data[:,2], X4:training_data[:,3], X5:training_data[:,4]})
#                print(predict_y)
#                mse = tf.reduce_mean(tf.square((predict_y-testing_label)))
#                print("MSE: %.4f" % sess.run(mse))
#    predict_y = sess.run(pred,feed_dict={X:training_data[:,0], X2:training_data[:,1], X3:training_data[:,2], X4:training_data[:,3], X5:training_data[:,4]})
#    a=[]
#    for i in range(len(predict_y)):
#        a.append([predict_y[i],training_label[i][0]])
#    a=np.array(a)
#    print(predict_y)
#    mse = tf.reduce_mean(tf.square((predict_y-testing_label)))
#    print("MSE: %.4f" % sess.run(mse))
#
#
##用来测试的
#test_X=np.array([[0.5,0.43,0.9,0.7,0.8],[0.22,0.34,0.13,0.89,0.64],[0.84,0.65,0.52,0.71,0.93]])
#test_Y=np.array([[10],[5],[1]])
#test_Y=np.matmul(test_X,[[2],[3],[2],[1],[4]])
#X_train=test_X
#Y_train=test_Y
#
#test_X=x_data[:3,:]
#
#
#################################################################################
#'''另一种靠谱的方法'''
#'''
#x_data=np.random.random([1000,3])
# 
##系数矩阵的shape必须是（3，1）。如果是（3，）会导致收敛效果差，猜测可能是y-y_label处形状不匹配
#y_data=np.matmul(x_data,[[2],[3],[2]])
# 
# 
#x=tf.placeholder(tf.float32,[None,3])
#y=tf.placeholder(tf.float32,[None,1])
#weight=tf.Variable(tf.random_normal([3,1]),dtype=tf.float32)
#'''
#x_data=test_X
#y_data=test_Y
#
#
#x=tf.placeholder(tf.float32,[None,5])
#y=tf.placeholder(tf.float32,[None,1])
#
#
#weight=tf.Variable(tf.random_normal([5,1]),dtype=tf.float32)
#
##tf.ones[1,1]，也可以写成tf.ones[1]，这样相当于标量，标量可以直接与矩阵相加
#bias=tf.Variable(tf.ones([1]),dtype=tf.float32)
#
#
#y_label=tf.add(tf.matmul(x,weight),bias)
#loss=tf.reduce_mean(tf.square(y-y_label))
#train=tf.train.GradientDescentOptimizer(0.5).minimize(loss)
#
#
#with tf.Session() as sess:
#    #变量初始化，目的是给Graph上的图中的变量初始化。
#    sess.run(tf.global_variables_initializer())
#    for i in range(1000):
#        sess.run(train,feed_dict={x:x_data,y:y_data})
#        if(i%100==0):
#            a=sess.run(y_label,feed_dict={x:x_data,y:y_data})
#    print(sess.run(weight),sess.run(bias))









