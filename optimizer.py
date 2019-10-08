import cma
import tf_test
from sklearn.preprocessing import StandardScaler
import numpy as np
import random
import regular as re
import tkinter as tk
import tkinter.messagebox 
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
random.seed(16)

###############################################################################
'''optimizer interface'''
def axis_move(current_axis):
    current_axis[1]+=1
    current_axis[2]+=1
    if current_axis[1] >= 8:
        current_axis[1] = 0
        current_axis[0]+=2
    return current_axis

def quit_interface(parameter):
    fix_vector[10]=parameter  
    frame11.pack_forget()
    root.destroy()

def fix_price(trigger):
    fix_vector[9]=trigger
    frame10.pack_forget()
    frame11.pack()
    frame=frame11
    root.title('fix the price?')
    
    text= tk.Label(frame, text='the price has to be fixed')
    text.pack( side = tk.TOP)
    
    botton1=tk.Button(frame,text='start the optimizer', command=lambda:quit_interface('1'))
    botton1.pack(side = tk.BOTTOM)

def fix_gear_number(trigger):
    fix_vector[8]=trigger
    frame9.pack_forget()
    frame10.pack()
    frame=frame10
    root.title('fix the gear number?')
    
    text= tk.Label(frame, text='fix the gear number?')
    text.pack( side = tk.TOP)
    
    botton1=tk.Button(frame,text='yes', command=lambda:fix_price('1'))
    botton1.pack(side = tk.LEFT)
    
    botton2=tk.Button(frame,text='no', command=lambda:fix_price('0'))
    botton2.pack(side = tk.RIGHT)


def fix_saddle_color(trigger):
    fix_vector[7]=trigger
    frame8.pack_forget()
    frame9.pack()
    frame=frame9
    root.title('fix the saddle color?')
    
    text= tk.Label(frame, text='fix the saddle color?')
    text.pack( side = tk.TOP)
    
    botton1=tk.Button(frame,text='yes', command=lambda:fix_gear_number('1'))
    botton1.pack(side = tk.LEFT)
    
    botton2=tk.Button(frame,text='no', command=lambda:fix_gear_number('0'))
    botton2.pack(side = tk.RIGHT)

def fix_bike_type(trigger):
    fix_vector[6]=trigger
    frame7.pack_forget()
    frame8.pack()
    frame=frame8
    root.title('fix the bike type?')
    
    text= tk.Label(frame, text='fix the bike type?')
    text.pack( side = tk.TOP)
    
    botton1=tk.Button(frame,text='yes', command=lambda:fix_saddle_color('1'))
    botton1.pack(side = tk.LEFT)
    
    botton2=tk.Button(frame,text='no', command=lambda:fix_saddle_color('0'))
    botton2.pack(side = tk.RIGHT)


def fix_front_carrier(trigger):
    fix_vector[5]=trigger
    frame6.pack_forget()
    frame7.pack()
    frame=frame7
    root.title('fix the front carrier?')
    
    text= tk.Label(frame, text='fix the front carrier?')
    text.pack( side = tk.TOP)
    
    botton1=tk.Button(frame,text='yes', command=lambda:fix_bike_type('1'))
    botton1.pack(side = tk.LEFT)
    
    botton2=tk.Button(frame,text='no', command=lambda:fix_bike_type('0'))
    botton2.pack(side = tk.RIGHT)

def fix_engine_position(trigger):
    fix_vector[4]=trigger
    frame5.pack_forget()
    frame6.pack()
    frame=frame6
    root.title('fix the engine position?')
    
    text= tk.Label(frame, text='fix the engine position?')
    text.pack( side = tk.TOP)
    
    botton1=tk.Button(frame,text='yes', command=lambda:fix_front_carrier('1'))
    botton1.pack(side = tk.LEFT)
    
    botton2=tk.Button(frame,text='no', command=lambda:fix_front_carrier('0'))
    botton2.pack(side = tk.RIGHT)


def fix_battery_position(trigger):
    fix_vector[3]=trigger
    frame4.pack_forget()
    frame5.pack()
    frame=frame5
    root.title('fix the battery position?')
    
    text= tk.Label(frame, text='fix the battery position?')
    text.pack( side = tk.TOP)
    
    botton1=tk.Button(frame,text='yes', command=lambda:fix_engine_position('1'))
    botton1.pack(side = tk.LEFT)
    
    botton2=tk.Button(frame,text='no', command=lambda:fix_engine_position('0'))
    botton2.pack(side = tk.RIGHT)

    
def fix_frame_type(trigger):
    fix_vector[2]=trigger
    frame3.pack_forget()
    frame4.pack()
    frame=frame4
    root.title('fix the frame type?')
    
    text= tk.Label(frame, text='fix the frame type?')
    text.pack( side = tk.TOP)
    
    botton1=tk.Button(frame,text='yes', command=lambda:fix_battery_position('1'))
    botton1.pack(side = tk.LEFT)
    
    botton2=tk.Button(frame,text='no', command=lambda:fix_battery_position('0'))
    botton2.pack(side = tk.RIGHT)


def fix_wheel_colors(trigger):
    fix_vector[1]=trigger
    frame2.pack_forget()
    frame3.pack()
    frame=frame3
    root.title('fix the wheel colors?')
    
    text= tk.Label(frame, text="fix the wheel colors?")
    text.pack( side = tk.TOP)
    
    botton1=tk.Button(frame,text='yes', command=lambda:fix_frame_type('1'))
    botton1.pack(side = tk.LEFT)
    
    botton2=tk.Button(frame,text='no', command=lambda:fix_frame_type('0'))
    botton2.pack(side = tk.RIGHT)

def fix_frame_colors(trigger):
    fix_vector[0]=trigger
    frame1.pack_forget()
    frame2.pack()
    frame=frame2
    root.title('fix the bike brand?')
    
    text= tk.Label(frame, text="fix the frame color?")
    text.pack( side = tk.TOP)
    
    botton1=tk.Button(frame,text='yes', command=lambda:fix_wheel_colors('1'))
    botton1.pack(side = tk.LEFT)
    
    botton2=tk.Button(frame,text='no', command=lambda:fix_wheel_colors('0'))
    botton2.pack(side = tk.RIGHT)

def fix_brands():
    frame1.pack()
    frame=frame1
    root.title('fix the bike brand?')
    
    text= tk.Label(frame, text="fix the bike brand?")
    text.pack( side = tk.TOP)
    
    botton1=tk.Button(frame,text='yes', command=lambda:fix_frame_colors('1'))
    botton1.pack(side = tk.LEFT)
    
    botton2=tk.Button(frame,text='no', command=lambda:fix_frame_colors('0'))
    botton2.pack(side = tk.RIGHT)



###############################################################################
def Hyperparametric_optimization(x):
    print(int(x[0]),int(x[1]),int(x[2]),int(x[3]),x[4])
    num_nodes_L1=int(x[0])
    num_nodes_L2=int(x[1])
    k=int(x[2])
    num_for_each_k=int(x[3])
    step_size=x[4]
    y = be.objective_function(feature_vectors,labels,bike_names,num_nodes_L1,num_nodes_L2,k,num_for_each_k,step_size)
    return y

def define_range(en_dic):
    dic_range=[]
    for i in range(len(en_dic)):
        if i == len(en_dic)-1:
            dic_range.append(float('inf'))
            break
        dic_range.append(len(en_dic[i])-1)
    return dic_range

def decode_the_features(features):
    decoder=[]
    for i in range(len(features)):
        if i == len(features)-1:
            break
        for j in range(len(en_dic[i])):
            if int(features[i]) == j:
                decoder.append(en_dic[i][j][0])
    decoder.append(en_dic[-1][int(features[-1])][0]*100)
    return decoder
                

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

def evaluate_e_bike(evaluation_data):
    encode_e_d=[]
    for i in range(len(en_dic)):
        for j in range(len(en_dic[i])):
            if j == evaluation_data[0][i]:
                for z in range(len(en_dic[i][j][1])):
                    encode_e_d.append(en_dic[i][j][1][z])    
    evaluation_data=[encode_e_d]
    bike_evaluation_result=tf_test.predict_with_the_trained_model(evaluation_data)
    print('bike score after optimized: '+str(bike_evaluation_result[0][0]))
    return bike_evaluation_result[0][0]

#只能通过解码的方式来编辑
#已经考虑到了特征界限的问题了
def certain_bike_optimization(evaluation_data):
    for i in range(len(evaluation_data)):
        if fix_vector[i] == '1':
            evaluation_data[i]=opt_fea_vec[i]
    for i in range(len(evaluation_data)):
        evaluation_data[i]=int(evaluation_data[i])
        if i == len(evaluation_data)-1:
            break
        else:
            if evaluation_data[i] < 0:
                evaluation_data[i]=0
            if evaluation_data[i] > dic_range[i]:
                evaluation_data[i] = dic_range[i]
    if evaluation_data[-1] < min_price:
        evaluation_data[-1] = min_price
    evaluation_data=[evaluation_data]
    y=(-1)*evaluate_e_bike(evaluation_data)
    opt_score_tracer.append(-y)
    return y




try:
    min_price=opt_fea_vec[-1]
    ftarget=0.7
    dic_range=define_range(en_dic)
    #界面
    fix_vector=[0,0,0,0,0,0,0,0,0,0,0]
    root = tk.Tk()
    frame1=tk.Frame(root)
    frame2=tk.Frame(root)
    frame3=tk.Frame(root)
    frame4=tk.Frame(root)
    frame5=tk.Frame(root)
    frame6=tk.Frame(root)
    frame7=tk.Frame(root)
    frame8=tk.Frame(root)
    frame9=tk.Frame(root)
    frame10=tk.Frame(root)
    frame11=tk.Frame(root)
    fix_brands()
    root.mainloop()
except:
    print('no feature vector available, pls run interface.py first.')

opt_score_tracer=[]
if fix_vector[-1]==0:
    print('invalid feature vector, pls reload again')
if fix_vector[-1]!=0:
    res = cma.fmin(certain_bike_optimization,opt_fea_vec,0.2,options={'ftarget':-ftarget,'popsize':10})
    decoder=decode_the_features(res[0])
    print('')
    print('AFTER OPTIMIZED REPORT')
    print('brand: '+str(re.regular_color(decoder[0])))
    print('frame color: '+str(re.regular_color(decoder[1])))
    print('wheel color: '+str(re.regular_color(decoder[2])))
    print('frame type: '+str(re.regular_color(decoder[3])))
    print('battery position: '+str(re.regular_color(decoder[4])))
    print('engine position: '+str(re.regular_color(decoder[5])))
    print('front carrier: '+str(decoder[6]))
    print('bike type: '+str(re.regular_color(decoder[7])))
    print('saddle color: '+str(re.regular_color(decoder[8])))
    print('gear number: '+str(re.regular_color(decoder[9])))
    print('price: '+str(decoder[10]))
    print('final score = '+str(-res[1]))



for i in range(len(opt_score_tracer)):
    if opt_score_tracer[i] == max(opt_score_tracer):
        optimal_score=[max(opt_score_tracer)]
        optimal_population=[i]
plt.figure()
plt.scatter(range(len(opt_score_tracer)),opt_score_tracer,marker='v',color='orange')
plt.scatter(optimal_population,optimal_score,marker='v',color='red',label='optimal design')
plt.xlabel('population')
plt.ylabel('design score')
plt.legend()
plt.show()

