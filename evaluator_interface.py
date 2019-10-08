import tkinter as tk
import tkinter.messagebox 
from PIL import Image, ImageTk
from sklearn.preprocessing import StandardScaler
import numpy as np
import tf_test


def resize(w, h, w_box, h_box, pil_image):   
    f1 = 1.0*w_box/w
    f2 = 1.0*h_box/h  
    factor = min([f1, f2])  

    width = int(w*factor)  
    height = int(h*factor)  
    return pil_image.resize((width, height), Image.ANTIALIAS)  
    

def format_a_photo(photo_name,w_box,h_box): 
    image = Image.open(photo_name)
    w, h = image.size  
    image=resize( w, h, w_box, h_box, image)  
    photo = ImageTk.PhotoImage(image)
    return photo

def axis_move(current_axis):
    current_axis[1]+=1
    current_axis[2]+=1
    if current_axis[1] >= 8:
        current_axis[1] = 0
        current_axis[0]+=2
    return current_axis

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

def creat_test_feature_vector(a):
    feature_vector=[]
    for i in range(len(en_dic)):
        for j in range(len(en_dic[i])):
            if a[i] == en_dic[i][j][0]:
                feature_vector.append(en_dic[i][j][1])
                break
            if j == len(en_dic[i])-1:
                print('did not find the match feature, pls retrain the model')
    unpacked_fea_vec=[]
    for i in range(len(feature_vector)):
        for j in range(len(feature_vector[i])):
            unpacked_fea_vec.append(feature_vector[i][j])
    feature_vector=[unpacked_fea_vec]
    #feature_vector[0].append(a[-1])
    opt_fea_vec=feature_vector
#    feature_vector=encode_test_data(uncode_feature_vectors,feature_vector)
    return feature_vector,opt_fea_vec[0]
###############################################################################
def quit_interface(parameter):
    create_vector[10]=parameter/100
    frame11.pack_forget()
    root.destroy()

def quit_reporter():
    frame1.pack_forget()
    root.destroy()


def price(parameter):
    create_vector[9]=parameter
    frame10.pack_forget()
    frame11.pack()
    frame=frame11
    root.title('insert recommend sales price')
    text= tk.Label(frame, text="insert recommend sales price")
    text.pack( side = tk.TOP)
    text_box=tk.Entry(frame, bd =5)
    text_box.pack(side = tk.BOTTOM)
    button=tk.Button(frame,text='click to submit', command=lambda:quit_interface(int(text_box.get())))
    button.pack(side = tk.BOTTOM)
    
def gear_number(parameter):
    #这一部分必须改
    create_vector[8]=parameter
    frame9.pack_forget()
    frame10.pack()
    frame=frame10
    root.title('select the gears number')
    current_axis=[0,0,0]
    
    tk.Button(frame,text='1', command=lambda:price('1~3')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    tk.Button(frame,text='2', command=lambda:price('1~3')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    tk.Button(frame,text='3', command=lambda:price('1~3')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    tk.Button(frame,text='4', command=lambda:price('4~10')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    tk.Button(frame,text='5', command=lambda:price('4~10')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    tk.Button(frame,text='6', command=lambda:price('4~10')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    tk.Button(frame,text='7', command=lambda:price('4~10')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    tk.Button(frame,text='8', command=lambda:price('4~10')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    tk.Button(frame,text='9', command=lambda:price('4~10')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    tk.Button(frame,text='10', command=lambda:price('4~10')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    tk.Button(frame,text='11', command=lambda:price('11~20')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    tk.Button(frame,text='12', command=lambda:price('11~20')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    tk.Button(frame,text='14', command=lambda:price('11~20')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    tk.Button(frame,text='17', command=lambda:price('11~20')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    tk.Button(frame,text='20', command=lambda:price('11~20')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    tk.Button(frame,text='22', command=lambda:price('22~30')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    tk.Button(frame,text='24', command=lambda:price('22~30')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    tk.Button(frame,text='27', command=lambda:price('22~30')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    tk.Button(frame,text='30', command=lambda:price('22~30')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    tk.Button(frame,text='traploos', command=lambda:price('traploos')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)

  
    
def saddle_color(parameter):
    w_box = 200
    h_box = 200
    #这一部分必须改
    create_vector[7]=parameter
    frame8.pack_forget()
    frame9.pack()
    frame=frame9
    root.title('select the saddle color')
    current_axis=[0,0,0]
    photos=[]
    photos.append(format_a_photo( "./encode_dictionary/saddle_color/black and white.jpg",w_box,h_box))
    photos.append(format_a_photo( "./encode_dictionary/saddle_color/black.jpg",w_box,h_box))
    photos.append(format_a_photo( "./encode_dictionary/saddle_color/brown.jpg",w_box,h_box))
    photos.append(format_a_photo( "./encode_dictionary/saddle_color/electric green.jpg",w_box,h_box))
    photos.append(format_a_photo( "./encode_dictionary/saddle_color/electric red.jpg",w_box,h_box))
    photos.append(format_a_photo( "./encode_dictionary/saddle_color/gray.jpg",w_box,h_box))
    photos.append(format_a_photo( "./encode_dictionary/saddle_color/orange.jpg",w_box,h_box))
    photos.append(format_a_photo( "./encode_dictionary/saddle_color/sky blue.jpg",w_box,h_box))
    photos.append(format_a_photo( "./encode_dictionary/saddle_color/white.jpg",w_box,h_box))

    
    label_img1 = tk.Label(frame, image = photos[current_axis[2]]).grid(row=current_axis[0], column=current_axis[1])
    Button1=tk.Button(frame,text='black and white', command=lambda:gear_number('b/w')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    label_img2 = tk.Label(frame, image = photos[current_axis[2]]).grid(row=current_axis[0], column=current_axis[1])
    Button2=tk.Button(frame,text='black', command=lambda:gear_number('b')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    label_img3 = tk.Label(frame, image = photos[current_axis[2]]).grid(row=current_axis[0], column=current_axis[1])
    Button3=tk.Button(frame,text='brown', command=lambda:gear_number('BR')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    label_img4 = tk.Label(frame, image = photos[current_axis[2]]).grid(row=current_axis[0], column=current_axis[1])
    Button4=tk.Button(frame,text='electric green', command=lambda:gear_number('eG')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    label_img5 = tk.Label(frame, image = photos[current_axis[2]]).grid(row=current_axis[0], column=current_axis[1])
    Button5=tk.Button(frame,text='electric red', command=lambda:gear_number('er')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    label_img6 = tk.Label(frame, image = photos[current_axis[2]]).grid(row=current_axis[0], column=current_axis[1])
    Button6=tk.Button(frame,text='gray', command=lambda:gear_number('g')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    label_img7 = tk.Label(frame, image = photos[current_axis[2]]).grid(row=current_axis[0], column=current_axis[1])
    Button7=tk.Button(frame,text='orange', command=lambda:gear_number('o')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    label_img8 = tk.Label(frame, image = photos[current_axis[2]]).grid(row=current_axis[0], column=current_axis[1])
    Button8=tk.Button(frame,text='sky blue', command=lambda:gear_number('skB')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    label_img9 = tk.Label(frame, image = photos[current_axis[2]]).grid(row=current_axis[0], column=current_axis[1])
    Button9=tk.Button(frame,text='white', command=lambda:gear_number('w')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    label_img1.pack()
    Button1.pack()
    label_img2.pack()
    Button2.pack()
    label_img3.pack()
    Button3.pack()
    label_img4.pack()
    Button4.pack()
    label_img5.pack()
    Button5.pack()
    label_img6.pack()
    Button6.pack()
    label_img7.pack()
    Button7.pack()
    label_img8.pack()
    Button8.pack()
    label_img9.pack()
    Button9.pack()

    

    
def bike_type(parameter):
    #这一部分必须改
    create_vector[6]=parameter
    frame7.pack_forget()
    frame8.pack()
    frame=frame8
    root.title('select the bike type')
    current_axis=[0,0,0]
    
    tk.Button(frame,text='Extra low Entry', command=lambda:saddle_color('Extra lage instap')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    tk.Button(frame,text='Tandem', command=lambda:saddle_color('Tandem')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    tk.Button(frame,text='Others', command=lambda:saddle_color('Onbekend')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    tk.Button(frame,text='Mountainbike', command=lambda:saddle_color('Mountainbike')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    tk.Button(frame,text='Race / sport bike', command=lambda:saddle_color('Race / sportfiets')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    tk.Button(frame,text='Compact', command=lambda:saddle_color('Compact')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    tk.Button(frame,text='Folding bicycle', command=lambda:saddle_color('Vouwfiets')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    tk.Button(frame,text='Gentlemen', command=lambda:saddle_color('Heren')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    tk.Button(frame,text='Ladys', command=lambda:saddle_color('Dames')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    tk.Button(frame,text='Cargo bike', command=lambda:saddle_color('Bakfiets')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    tk.Button(frame,text='Speed pedelec (45km/h)', command=lambda:saddle_color('Speedpedelec (45km/h)')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)


    
def front_carrier(parameter):
    w_box = 200
    h_box = 200
    #这一部分必须改
    create_vector[5]=parameter
    frame6.pack_forget()
    frame7.pack()
    frame=frame7
    root.title('does the bike have a front carrier?')
    current_axis=[0,0,0]
    photos=[]
    photos.append(format_a_photo( "./encode_dictionary/front_carrier/with front carrier.jpg",w_box,h_box))
    photos.append(format_a_photo( "./encode_dictionary/front_carrier/without front carrier.jpg",w_box,h_box))
    
    label_img1 = tk.Label(frame, image = photos[current_axis[2]]).grid(row=current_axis[0], column=current_axis[1])
    Button1=tk.Button(frame,text='with front carrier', command=lambda:bike_type('y')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    label_img2 = tk.Label(frame, image = photos[current_axis[2]]).grid(row=current_axis[0], column=current_axis[1])
    Button2=tk.Button(frame,text='without front carrier', command=lambda:bike_type('n')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    label_img1.pack()
    Button1.pack()
    label_img2.pack()
    Button2.pack()


    
def engine_position(parameter):
    w_box = 200
    h_box = 200
    #这一部分必须改
    create_vector[4]=parameter
    frame5.pack_forget()
    frame6.pack()
    frame=frame6
    root.title('select the engine position')
    current_axis=[0,0,0]
    photos=[]
    photos.append(format_a_photo( "./encode_dictionary/engine position/back wheel engine.jpg",w_box,h_box))
    photos.append(format_a_photo( "./encode_dictionary/engine position/front wheel engine.jpg",w_box,h_box))
    photos.append(format_a_photo( "./encode_dictionary/engine position/mid engine.jpg",w_box,h_box))
    
    label_img1 = tk.Label(frame, image = photos[current_axis[2]]).grid(row=current_axis[0], column=current_axis[1])
    Button1=tk.Button(frame,text='back wheel engine', command=lambda:front_carrier('In het achterwiel')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    label_img2 = tk.Label(frame, image = photos[current_axis[2]]).grid(row=current_axis[0], column=current_axis[1])
    Button2=tk.Button(frame,text='front wheel engine', command=lambda:front_carrier('In het voorwiel')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    label_img3 = tk.Label(frame, image = photos[current_axis[2]]).grid(row=current_axis[0], column=current_axis[1])
    Button3=tk.Button(frame,text='mid engine', command=lambda:front_carrier('In het midden')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    label_img1.pack()
    Button1.pack()
    label_img2.pack()
    Button2.pack()
    label_img3.pack()
    Button3.pack()
    
def battery_position(parameter):
    w_box = 200
    h_box = 200
    #这一部分必须改
    create_vector[3]=parameter
    frame4.pack_forget()
    frame5.pack()
    frame=frame5
    root.title('select the battery position')
    current_axis=[0,0,0]
    photos=[]
    photos.append(format_a_photo( "./encode_dictionary/battery position/In the frame.jpg",w_box,h_box))
    photos.append(format_a_photo( "./encode_dictionary/battery position/Under the luggage carrier.jpg",w_box,h_box))
    photos.append(format_a_photo( "./encode_dictionary/battery position/Visible on the frame.jpg",w_box,h_box))
    
    label_img1 = tk.Label(frame, image = photos[current_axis[2]]).grid(row=current_axis[0], column=current_axis[1])
    Button1=tk.Button(frame,text='In the frame', command=lambda:engine_position('In het frame weggewerkt')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    label_img2 = tk.Label(frame, image = photos[current_axis[2]]).grid(row=current_axis[0], column=current_axis[1])
    Button2=tk.Button(frame,text='Under the luggage carrier', command=lambda:engine_position('Onder de bagagedrager')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    label_img3 = tk.Label(frame, image = photos[current_axis[2]]).grid(row=current_axis[0], column=current_axis[1])
    Button3=tk.Button(frame,text='Visible on the frame', command=lambda:engine_position('Aan het frame zichtbaar')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    label_img1.pack()
    Button1.pack()
    label_img2.pack()
    Button2.pack()
    label_img3.pack()
    Button3.pack()


   
def frame_types(parameter):
    w_box = 200
    h_box = 200
    #这一部分必须改
    create_vector[2]=parameter
    frame3.pack_forget()
    frame4.pack()
    frame=frame4
    root.title('select the frame type')
    current_axis=[0,0,0]
    photos=[]
    photos.append(format_a_photo( "./encode_dictionary/frame_type/basket.jpg",w_box,h_box))
    photos.append(format_a_photo( "./encode_dictionary/frame_type/double seats.jpg",w_box,h_box))
    photos.append(format_a_photo( "./encode_dictionary/frame_type/female double thin thick.jpg",w_box,h_box))
    photos.append(format_a_photo( "./encode_dictionary/frame_type/female double thin thin.jpg",w_box,h_box))
    photos.append(format_a_photo( "./encode_dictionary/frame_type/female single thick.jpg",w_box,h_box))
    photos.append(format_a_photo( "./encode_dictionary/frame_type/female single thin.jpg",w_box,h_box))
    photos.append(format_a_photo( "./encode_dictionary/frame_type/fold.jpg",w_box,h_box))
    photos.append(format_a_photo( "./encode_dictionary/frame_type/male double thin thin.jpg",w_box,h_box))
    photos.append(format_a_photo( "./encode_dictionary/frame_type/male single thick thick.jpg",w_box,h_box))
    photos.append(format_a_photo( "./encode_dictionary/frame_type/male single thin thick.jpg",w_box,h_box))
    photos.append(format_a_photo( "./encode_dictionary/frame_type/male single thin thin.jpg",w_box,h_box))
    photos.append(format_a_photo( "./encode_dictionary/frame_type/mountain or race.jpg",w_box,h_box))
    photos.append(format_a_photo( "./encode_dictionary/frame_type/special type.jpg",w_box,h_box))
    photos.append(format_a_photo( "./encode_dictionary/frame_type/unrecognized pic.jpg",w_box,h_box))

    
    label_img1 = tk.Label(frame, image = photos[current_axis[2]]).grid(row=current_axis[0], column=current_axis[1])
    Button1=tk.Button(frame,text='basket', command=lambda:battery_position('bas')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    label_img2 = tk.Label(frame, image = photos[current_axis[2]]).grid(row=current_axis[0], column=current_axis[1])
    Button2=tk.Button(frame,text='double seats', command=lambda:battery_position('dou')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    label_img3 = tk.Label(frame, image = photos[current_axis[2]]).grid(row=current_axis[0], column=current_axis[1])
    Button3=tk.Button(frame,text='female double thin thick', command=lambda:battery_position('ddxc')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    label_img4 = tk.Label(frame, image = photos[current_axis[2]]).grid(row=current_axis[0], column=current_axis[1])
    Button4=tk.Button(frame,text='female double thin thin', command=lambda:battery_position('ddxx')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    label_img5 = tk.Label(frame, image = photos[current_axis[2]]).grid(row=current_axis[0], column=current_axis[1])
    Button5=tk.Button(frame,text='female single thick', command=lambda:battery_position('dsc')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    label_img6 = tk.Label(frame, image = photos[current_axis[2]]).grid(row=current_axis[0], column=current_axis[1])
    Button6=tk.Button(frame,text='female single thin', command=lambda:battery_position('dsx')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    label_img7 = tk.Label(frame, image = photos[current_axis[2]]).grid(row=current_axis[0], column=current_axis[1])
    Button7=tk.Button(frame,text='fold', command=lambda:battery_position('f')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    label_img8 = tk.Label(frame, image = photos[current_axis[2]]).grid(row=current_axis[0], column=current_axis[1])
    Button8=tk.Button(frame,text='male double thin thin', command=lambda:battery_position('hdxx')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    label_img9 = tk.Label(frame, image = photos[current_axis[2]]).grid(row=current_axis[0], column=current_axis[1])
    Button9=tk.Button(frame,text='male single thick thick', command=lambda:battery_position('hscc')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    label_img10 = tk.Label(frame, image = photos[current_axis[2]]).grid(row=current_axis[0], column=current_axis[1])
    Button10=tk.Button(frame,text='male single thin thick', command=lambda:battery_position('hsxc')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    label_img11 = tk.Label(frame, image = photos[current_axis[2]]).grid(row=current_axis[0], column=current_axis[1])
    Button11=tk.Button(frame,text='male single thin thin', command=lambda:battery_position('hsxx')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    label_img12 = tk.Label(frame, image = photos[current_axis[2]]).grid(row=current_axis[0], column=current_axis[1])
    Button12=tk.Button(frame,text='mountain or race', command=lambda:battery_position('mdxc')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    label_img13 = tk.Label(frame, image = photos[current_axis[2]]).grid(row=current_axis[0], column=current_axis[1])
    Button13=tk.Button(frame,text='special type', command=lambda:battery_position('sp')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    label_img14 = tk.Label(frame, image = photos[current_axis[2]]).grid(row=current_axis[0], column=current_axis[1])
    Button14=tk.Button(frame,text='unrecognized pic', command=lambda:battery_position('st')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)

    
    label_img1.pack()
    Button1.pack()
    label_img2.pack()
    Button2.pack()
    label_img3.pack()
    Button3.pack()
    label_img4.pack()
    Button4.pack()
    label_img5.pack()
    Button5.pack()
    label_img6.pack()
    Button6.pack()
    label_img7.pack()
    Button7.pack()
    label_img8.pack()
    Button8.pack()
    label_img9.pack()
    Button9.pack()
    label_img10.pack()
    Button10.pack()
    label_img11.pack()
    Button11.pack()
    label_img12.pack()
    Button12.pack()
    label_img13.pack()
    Button13.pack()
    label_img14.pack()
    Button14.pack()

    
def wheel_colors(parameter):
    w_box = 200
    h_box = 200
    #这一部分必须改
    create_vector[1]=parameter
    frame2.pack_forget()
    frame3.pack()
    frame=frame3
    root.title('select the wheel color')
    current_axis=[0,0,0]
    photos=[]
    photos.append(format_a_photo( "./encode_dictionary/wheel_colors/black and metal.jpg",w_box,h_box))
    photos.append(format_a_photo( "./encode_dictionary/wheel_colors/black and orange.jpg",w_box,h_box))
    photos.append(format_a_photo( "./encode_dictionary/wheel_colors/black and white.jpg",w_box,h_box))
    photos.append(format_a_photo( "./encode_dictionary/wheel_colors/black and yellow.jpg",w_box,h_box))
    photos.append(format_a_photo( "./encode_dictionary/wheel_colors/black.jpg",w_box,h_box))
    photos.append(format_a_photo( "./encode_dictionary/wheel_colors/gray.jpg",w_box,h_box))
    photos.append(format_a_photo( "./encode_dictionary/wheel_colors/orange.jpg",w_box,h_box))
    photos.append(format_a_photo( "./encode_dictionary/wheel_colors/white.jpg",w_box,h_box))
    
    label_img1 = tk.Label(frame, image = photos[current_axis[2]]).grid(row=current_axis[0], column=current_axis[1])
    Button1=tk.Button(frame,text='black and metal', command=lambda:frame_types('b/m')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    label_img2 = tk.Label(frame, image = photos[current_axis[2]]).grid(row=current_axis[0], column=current_axis[1])
    Button2=tk.Button(frame,text='black and orange', command=lambda:frame_types('b/o')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    label_img3 = tk.Label(frame, image = photos[current_axis[2]]).grid(row=current_axis[0], column=current_axis[1])
    Button3=tk.Button(frame,text='black and white', command=lambda:frame_types('b/w')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    label_img4 = tk.Label(frame, image = photos[current_axis[2]]).grid(row=current_axis[0], column=current_axis[1])
    Button4=tk.Button(frame,text='black and yellow', command=lambda:frame_types('b/y')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    label_img5 = tk.Label(frame, image = photos[current_axis[2]]).grid(row=current_axis[0], column=current_axis[1])
    Button5=tk.Button(frame,text='black', command=lambda:frame_types('b')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    label_img6 = tk.Label(frame, image = photos[current_axis[2]]).grid(row=current_axis[0], column=current_axis[1])
    Button6=tk.Button(frame,text='gray', command=lambda:frame_types('g')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    label_img7 = tk.Label(frame, image = photos[current_axis[2]]).grid(row=current_axis[0], column=current_axis[1])
    Button7=tk.Button(frame,text='orange', command=lambda:frame_types('o')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    label_img8 = tk.Label(frame, image = photos[current_axis[2]]).grid(row=current_axis[0], column=current_axis[1])
    Button8=tk.Button(frame,text='white', command=lambda:frame_types('w')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    label_img1.pack()
    Button1.pack()
    label_img2.pack()
    Button2.pack()
    label_img3.pack()
    Button3.pack()
    label_img4.pack()
    Button4.pack()
    label_img5.pack()
    Button5.pack()
    label_img6.pack()
    Button6.pack()
    label_img7.pack()
    Button7.pack()
    label_img8.pack()
    Button8.pack()
    

    
def frame_colors(parameter):
    w_box = 200
    h_box = 200
    #这一部分必须改
    create_vector[0]=parameter
    frame1.pack_forget()
    frame2.pack()
    frame=frame2
    root.title('select the frame color')
    current_axis=[0,0,0]
    photos=[]
    photos.append(format_a_photo( "./encode_dictionary/frame_colors/black and blue.jpg",w_box,h_box))
    photos.append(format_a_photo( "./encode_dictionary/frame_colors/black and champagne.jpg",w_box,h_box))
    photos.append(format_a_photo( "./encode_dictionary/frame_colors/black and electric green.jpg",w_box,h_box))
    photos.append(format_a_photo( "./encode_dictionary/frame_colors/black and gray.jpg",w_box,h_box))
    photos.append(format_a_photo( "./encode_dictionary/frame_colors/black and orange.jpg",w_box,h_box))
    photos.append(format_a_photo( "./encode_dictionary/frame_colors/black and red.jpg",w_box,h_box))
    photos.append(format_a_photo( "./encode_dictionary/frame_colors/black and white.jpg",w_box,h_box))
    photos.append(format_a_photo( "./encode_dictionary/frame_colors/black and yellow.jpg",w_box,h_box))
    photos.append(format_a_photo( "./encode_dictionary/frame_colors/black.jpg",w_box,h_box))
    photos.append(format_a_photo( "./encode_dictionary/frame_colors/blue.jpg",w_box,h_box))
    photos.append(format_a_photo( "./encode_dictionary/frame_colors/brown.jpg",w_box,h_box))
    photos.append(format_a_photo( "./encode_dictionary/frame_colors/champagne.jpg",w_box,h_box))
    photos.append(format_a_photo( "./encode_dictionary/frame_colors/dark blue.jpg",w_box,h_box))
    photos.append(format_a_photo( "./encode_dictionary/frame_colors/dark brown.jpg",w_box,h_box))
    photos.append(format_a_photo( "./encode_dictionary/frame_colors/dark gray.jpg",w_box,h_box))
    photos.append(format_a_photo( "./encode_dictionary/frame_colors/electric blue.jpg",w_box,h_box))
    photos.append(format_a_photo( "./encode_dictionary/frame_colors/electric green.jpg",w_box,h_box))
    photos.append(format_a_photo( "./encode_dictionary/frame_colors/gray and orange.jpg",w_box,h_box))
    photos.append(format_a_photo( "./encode_dictionary/frame_colors/gray and red.jpg",w_box,h_box))
    photos.append(format_a_photo( "./encode_dictionary/frame_colors/gray.jpg",w_box,h_box))
    photos.append(format_a_photo( "./encode_dictionary/frame_colors/green.jpg",w_box,h_box))
    photos.append(format_a_photo( "./encode_dictionary/frame_colors/light blue.jpg",w_box,h_box))
    photos.append(format_a_photo( "./encode_dictionary/frame_colors/light green.jpg",w_box,h_box))
    photos.append(format_a_photo( "./encode_dictionary/frame_colors/orange.jpg",w_box,h_box))
    photos.append(format_a_photo( "./encode_dictionary/frame_colors/red.jpg",w_box,h_box))
    photos.append(format_a_photo( "./encode_dictionary/frame_colors/sea blue.jpg",w_box,h_box))
    photos.append(format_a_photo( "./encode_dictionary/frame_colors/silver gray.jpg",w_box,h_box))
    photos.append(format_a_photo( "./encode_dictionary/frame_colors/sky blue.jpg",w_box,h_box))
    photos.append(format_a_photo( "./encode_dictionary/frame_colors/white.jpg",w_box,h_box))
    photos.append(format_a_photo( "./encode_dictionary/frame_colors/wine red.jpg",w_box,h_box))
    photos.append(format_a_photo( "./encode_dictionary/frame_colors/yellow gray.jpg",w_box,h_box))
    photos.append(format_a_photo( "./encode_dictionary/frame_colors/yellow.jpg",w_box,h_box))
    
    
    label_img1 = tk.Label(frame, image = photos[current_axis[2]]).grid(row=current_axis[0], column=current_axis[1])
    Button1=tk.Button(frame,text='black and blue', command=lambda:wheel_colors('b/B')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    label_img2 = tk.Label(frame, image = photos[current_axis[2]]).grid(row=current_axis[0], column=current_axis[1])
    Button2=tk.Button(frame,text='black and champagne', command=lambda:wheel_colors('b/cham')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    label_img3 = tk.Label(frame, image = photos[current_axis[2]]).grid(row=current_axis[0], column=current_axis[1])
    Button3=tk.Button(frame,text='black and electric green', command=lambda:wheel_colors('b/eG')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    label_img4 = tk.Label(frame, image = photos[current_axis[2]]).grid(row=current_axis[0], column=current_axis[1])
    Button4=tk.Button(frame,text='black and gray', command=lambda:wheel_colors('b/g')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    label_img5 = tk.Label(frame, image = photos[current_axis[2]]).grid(row=current_axis[0], column=current_axis[1])
    Button5=tk.Button(frame,text='black and orange', command=lambda:wheel_colors('b/o')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    label_img6 = tk.Label(frame, image = photos[current_axis[2]]).grid(row=current_axis[0], column=current_axis[1])
    Button6=tk.Button(frame,text='black and red', command=lambda:wheel_colors('b/r')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    label_img7 = tk.Label(frame, image = photos[current_axis[2]]).grid(row=current_axis[0], column=current_axis[1])
    Button7=tk.Button(frame,text='black and white', command=lambda:wheel_colors('b/w')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    label_img8 = tk.Label(frame, image = photos[current_axis[2]]).grid(row=current_axis[0], column=current_axis[1])
    Button8=tk.Button(frame,text='black and yellow', command=lambda:wheel_colors('b/y')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    label_img9 = tk.Label(frame, image = photos[current_axis[2]]).grid(row=current_axis[0], column=current_axis[1])
    Button9=tk.Button(frame,text='black', command=lambda:wheel_colors('b')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    label_img10 = tk.Label(frame, image = photos[current_axis[2]]).grid(row=current_axis[0], column=current_axis[1])
    Button10=tk.Button(frame,text='blue', command=lambda:wheel_colors('B')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    label_img11 = tk.Label(frame, image = photos[current_axis[2]]).grid(row=current_axis[0], column=current_axis[1])
    Button11=tk.Button(frame,text='brown', command=lambda:wheel_colors('BR')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    label_img12 = tk.Label(frame, image = photos[current_axis[2]]).grid(row=current_axis[0], column=current_axis[1])
    Button12=tk.Button(frame,text='champagne', command=lambda:wheel_colors('cham')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    label_img13 = tk.Label(frame, image = photos[current_axis[2]]).grid(row=current_axis[0], column=current_axis[1])
    Button13=tk.Button(frame,text='dark blue', command=lambda:wheel_colors('dB')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    label_img14 = tk.Label(frame, image = photos[current_axis[2]]).grid(row=current_axis[0], column=current_axis[1])
    Button14=tk.Button(frame,text='dark brown', command=lambda:wheel_colors('dBR')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    label_img15 = tk.Label(frame, image = photos[current_axis[2]]).grid(row=current_axis[0], column=current_axis[1])
    Button15=tk.Button(frame,text='dark gray', command=lambda:wheel_colors('dg')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    label_img16 = tk.Label(frame, image = photos[current_axis[2]]).grid(row=current_axis[0], column=current_axis[1])
    Button16=tk.Button(frame,text='electric blue', command=lambda:wheel_colors('eB')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    label_img17 = tk.Label(frame, image = photos[current_axis[2]]).grid(row=current_axis[0], column=current_axis[1])
    Button17=tk.Button(frame,text='electric green', command=lambda:wheel_colors('eG')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    label_img18 = tk.Label(frame, image = photos[current_axis[2]]).grid(row=current_axis[0], column=current_axis[1])
    Button18=tk.Button(frame,text='gray and orange', command=lambda:wheel_colors('g/o')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    label_img19 = tk.Label(frame, image = photos[current_axis[2]]).grid(row=current_axis[0], column=current_axis[1])
    Button19=tk.Button(frame,text='gray and red', command=lambda:wheel_colors('g/r')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    label_img20 = tk.Label(frame, image = photos[current_axis[2]]).grid(row=current_axis[0], column=current_axis[1])
    Button20=tk.Button(frame,text='gray', command=lambda:wheel_colors('g')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    label_img21 = tk.Label(frame, image = photos[current_axis[2]]).grid(row=current_axis[0], column=current_axis[1])
    Button21=tk.Button(frame,text='green', command=lambda:wheel_colors('G')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    label_img22 = tk.Label(frame, image = photos[current_axis[2]]).grid(row=current_axis[0], column=current_axis[1])
    Button22=tk.Button(frame,text='light blue', command=lambda:wheel_colors('lB')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    label_img23 = tk.Label(frame, image = photos[current_axis[2]]).grid(row=current_axis[0], column=current_axis[1])
    Button23=tk.Button(frame,text='light green', command=lambda:wheel_colors('lG')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    label_img24 = tk.Label(frame, image = photos[current_axis[2]]).grid(row=current_axis[0], column=current_axis[1])
    Button24=tk.Button(frame,text='orange', command=lambda:wheel_colors('o')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    label_img25 = tk.Label(frame, image = photos[current_axis[2]]).grid(row=current_axis[0], column=current_axis[1])
    Button25=tk.Button(frame,text='red', command=lambda:wheel_colors('r')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    label_img26 = tk.Label(frame, image = photos[current_axis[2]]).grid(row=current_axis[0], column=current_axis[1])
    Button26=tk.Button(frame,text='sea blue', command=lambda:wheel_colors('seaB')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    label_img27 = tk.Label(frame, image = photos[current_axis[2]]).grid(row=current_axis[0], column=current_axis[1])
    Button27=tk.Button(frame,text='silver gray', command=lambda:wheel_colors('sg')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    label_img28 = tk.Label(frame, image = photos[current_axis[2]]).grid(row=current_axis[0], column=current_axis[1])
    Button28=tk.Button(frame,text='sky blue', command=lambda:wheel_colors('skB')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    label_img29 = tk.Label(frame, image = photos[current_axis[2]]).grid(row=current_axis[0], column=current_axis[1])
    Button29=tk.Button(frame,text='white', command=lambda:wheel_colors('w')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    label_img30 = tk.Label(frame, image = photos[current_axis[2]]).grid(row=current_axis[0], column=current_axis[1])
    Button30=tk.Button(frame,text='wine red', command=lambda:wheel_colors('wr')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    label_img31 = tk.Label(frame, image = photos[current_axis[2]]).grid(row=current_axis[0], column=current_axis[1])
    Button31=tk.Button(frame,text='yellow gray', command=lambda:wheel_colors('yg')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    label_img32 = tk.Label(frame, image = photos[current_axis[2]]).grid(row=current_axis[0], column=current_axis[1])
    Button32=tk.Button(frame,text='yellow', command=lambda:wheel_colors('y')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    label_img1.pack()
    Button1.pack()
    label_img2.pack()
    Button2.pack()
    label_img3.pack()
    Button3.pack()
    label_img4.pack()
    Button4.pack()
    label_img5.pack()
    Button5.pack()
    label_img6.pack()
    Button6.pack()
    label_img7.pack()
    Button7.pack()
    label_img8.pack()
    Button8.pack()
    label_img9.pack()
    Button9.pack()
    label_img10.pack()
    Button10.pack()
    label_img11.pack()
    Button11.pack()
    label_img12.pack()
    Button12.pack()
    label_img13.pack()
    Button13.pack()
    label_img14.pack()
    Button14.pack()
    label_img15.pack()
    Button15.pack()
    label_img16.pack()
    Button16.pack()
    label_img17.pack()
    Button17.pack()
    label_img18.pack()
    Button18.pack()
    label_img19.pack()
    Button19.pack()
    label_img20.pack()
    Button20.pack()
    label_img21.pack()
    Button21.pack()
    label_img22.pack()
    Button22.pack()
    label_img23.pack()
    Button23.pack()
    label_img24.pack()
    Button24.pack()
    label_img25.pack()
    Button25.pack()
    label_img26.pack()
    Button26.pack()
    label_img27.pack()
    Button27.pack()
    label_img28.pack()
    Button28.pack()
    label_img29.pack()
    Button29.pack()
    label_img30.pack()
    Button30.pack()
    label_img31.pack()
    Button31.pack()
    label_img32.pack()
    Button32.pack()
    





def brands():
    frame1.pack()
    frame=frame1
    root.title('select the bike brand')
    current_axis=[0,0,0]
    tk.Button(frame,text='a-bike', command=lambda:frame_colors('a-bike')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    tk.Button(frame,text='altec', command=lambda:frame_colors('altec')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    tk.Button(frame,text='ampler-bikes', command=lambda:frame_colors('ampler-bikes')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)

    tk.Button(frame,text='amslod', command=lambda:frame_colors('amslod')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)

    tk.Button(frame,text='bakfiets', command=lambda:frame_colors('bakfiets')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    tk.Button(frame,text='batavus', command=lambda:frame_colors('batavus')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    tk.Button(frame,text='bikkel-bikes', command=lambda:frame_colors('bikkel-bikes')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    tk.Button(frame,text='bizobike', command=lambda:frame_colors('bizobike')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    tk.Button(frame,text='brinckers', command=lambda:frame_colors('brinckers')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    tk.Button(frame,text='brinke', command=lambda:frame_colors('brinke')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    tk.Button(frame,text='bsp-fietsen', command=lambda:frame_colors('bsp-fietsen')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    tk.Button(frame,text='btwin', command=lambda:frame_colors('btwin')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    tk.Button(frame,text='cannondale', command=lambda:frame_colors('cannondale')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    tk.Button(frame,text='clikebikes', command=lambda:frame_colors('clikebikes')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    tk.Button(frame,text='cortina', command=lambda:frame_colors('cortina')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    tk.Button(frame,text='cross', command=lambda:frame_colors('cross')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    tk.Button(frame,text='cube', command=lambda:frame_colors('cube')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    tk.Button(frame,text='dahon', command=lambda:frame_colors('dahon')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    tk.Button(frame,text='dutch-id', command=lambda:frame_colors('dutch-id')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    tk.Button(frame,text='e-bikez', command=lambda:frame_colors('e-bikez')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    tk.Button(frame,text='ebike-das-original', command=lambda:frame_colors('ebike-das-original')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    tk.Button(frame,text='eco-traveller', command=lambda:frame_colors('eco-traveller')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    tk.Button(frame,text='flyer', command=lambda:frame_colors('flyer')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    tk.Button(frame,text='gazelle', command=lambda:frame_colors('gazelle')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    tk.Button(frame,text='gepida', command=lambda:frame_colors('gepida')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    tk.Button(frame,text='giant', command=lambda:frame_colors('giant')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    tk.Button(frame,text='hercules', command=lambda:frame_colors('hercules')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    tk.Button(frame,text='ideal', command=lambda:frame_colors('ideal')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    tk.Button(frame,text='kalkhoff', command=lambda:frame_colors('kalkhoff')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    tk.Button(frame,text='klever', command=lambda:frame_colors('klever')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    tk.Button(frame,text='koga', command=lambda:frame_colors('koga')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    tk.Button(frame,text='merida', command=lambda:frame_colors('merida')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    tk.Button(frame,text='multicycle', command=lambda:frame_colors('multicycle')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    tk.Button(frame,text='oxford', command=lambda:frame_colors('oxford')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    tk.Button(frame,text='pegasus', command=lambda:frame_colors('pegasus')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    tk.Button(frame,text='popal', command=lambda:frame_colors('popal')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    tk.Button(frame,text='pro-e-bike', command=lambda:frame_colors('pro-e-bike')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    tk.Button(frame,text='puch', command=lambda:frame_colors('puch')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    tk.Button(frame,text='qwic', command=lambda:frame_colors('qwic')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    tk.Button(frame,text='rap', command=lambda:frame_colors('rap')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    tk.Button(frame,text='riese-und-muller', command=lambda:frame_colors('riese-und-muller')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    tk.Button(frame,text='rivel', command=lambda:frame_colors('rivel')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    tk.Button(frame,text='rose', command=lambda:frame_colors('rose')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    tk.Button(frame,text='scott', command=lambda:frame_colors('scott')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    tk.Button(frame,text='shinga', command=lambda:frame_colors('shinga')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    tk.Button(frame,text='sparta', command=lambda:frame_colors('sparta')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    tk.Button(frame,text='specialized', command=lambda:frame_colors('specialized')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    tk.Button(frame,text='spiked-cycles', command=lambda:frame_colors('spiked-cycles')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    tk.Button(frame,text='stella', command=lambda:frame_colors('stella')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    tk.Button(frame,text='stromer', command=lambda:frame_colors('stromer')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    tk.Button(frame,text='trek', command=lambda:frame_colors('trek')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    tk.Button(frame,text='trenergy', command=lambda:frame_colors('trenergy')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    tk.Button(frame,text='velo-de-ville', command=lambda:frame_colors('velo-de-ville')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    tk.Button(frame,text='victesse', command=lambda:frame_colors('victesse')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    tk.Button(frame,text='vogue-bike', command=lambda:frame_colors('vogue-bike')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    tk.Button(frame,text='votani', command=lambda:frame_colors('votani')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)
    
    tk.Button(frame,text='watt', command=lambda:frame_colors('watt')).grid(row=current_axis[0]+1, column=current_axis[1])
    current_axis=axis_move(current_axis)


create_vector=[0,0,0,0,0,0,0,0,0,0,0]

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
brands()
root.mainloop()

if create_vector[-1]==0:
    print('invalid feature vector, pls reload again')
if create_vector[-1]!=0:
    feature_vector,opt_fea_vec=creat_test_feature_vector(create_vector)
    bike_evaluation_result=tf_test.predict_with_the_trained_model(feature_vector)
    root = tk.Tk()
    frame1=tk.Frame(root)
    frame1.pack()
    frame=frame1
    root.title('Evaluator Report:')
    text= tk.Label(frame, text='Bike Scored:'+str(bike_evaluation_result[0][0]))
    text.pack( side = tk.TOP)
    '''
    text1= tk.Label(frame, text='You can run optimizer.py to optimize the bike design')
    text1.pack( side = tk.TOP)
    '''
    button=tk.Button(frame,text='OK', command=quit_reporter)
    button.pack(side = tk.BOTTOM)
    root.mainloop()

        
#先把evaluation_data和opt_fea_vec都斩断，方便optimizer分析
opt_breaker=[]
indexer=0
for i in range(len(en_dic)):
    for j in range(len(en_dic[i])):
        if opt_fea_vec[indexer+j] == 1:
            opt_breaker.append(j)
            break
    indexer+=len(en_dic[i])
opt_fea_vec=opt_breaker



