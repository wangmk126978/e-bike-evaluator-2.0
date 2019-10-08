import numpy as np
import pandas as pd
import time
import datetime
import google.auth
from google.cloud import bigquery
from google.cloud import bigquery_storage_v1beta1
import json
import re
import matplotlib.image as mpimg # mpimg 用于读取图片
from bs4 import BeautifulSoup as bs
import urllib
import csv
import os
import matplotlib.pyplot as plt
import matplotlib


'''基础工具箱'''
#写入txt文档
#file_name:'kangkang.txt',method:'zhuijia'/'fugai',content：list/variable
def kangkang_write_txt(file_name,method,content):
    if method == 'zhuijia':
        m='a'
    if method == 'fugai':
        m='w'
    f=open(file_name,m)
    f.write(content)
    f.close()
    
#将json格式的数据转换成dictionary
#data_need_to_be_transform
def kangkang_json_2_dic(data_need_to_be_transform):
    json2dic=json.loads(data_need_to_be_transform)
    return json2dic

#读取dictionary中的一类attribute,并以list的形式输出
#att_name: 'values'
def kangkang_find_a_attribute_from_dic(dic,att_name):
    attribute=list(dic[att_name])
    return attribute

#在桌面批量建立文件夹，根据名称list和根文件夹名
#folder_names:一个由名字组成的list, root_folder_name:根文件夹名称
def kangkang_creat_pics_folder(folder_names,root_folder_name):
    for i in range(len(folder_names)):
        folder_name=folder_names[i][0]
        file = "H:\\Desktop\\"+str(root_folder_name)+"\\"+folder_name
        os.makedirs(file)   

#下载图片到指定地址
#pic_src：'https://www.fietsenwinkel.nl/pub/media/logo/stores/1/fietsenwinkel_nl_logo.png' , address:'H:\\Desktop\\' , pic_name: 'haha.jpg'
def kangkang_download_pic(pic_src,address,pic_name):
    urllib.request.urlretrieve(pic_src,address+pic_name)
    
#将list转成data frame,最多3类，多了需要自己改写
#target_list:多行的数据内容, df_attributes:1xn的数组，保存着每行数据内容的名称
def kangkang_transform_list_to_dataframe(target_list,df_attributes):
    dataframe='error'
    if len(df_attributes) == 1 and len(target_list) == 1:
        dataframe=pd.DataFrame({df_attributes[0]:target_list[0]})
    if len(df_attributes) == 2 and len(target_list) == 2:
        dataframe=pd.DataFrame({df_attributes[0]:target_list[0],df_attributes[1]:target_list[1]})
    if len(df_attributes) == 3 and len(target_list) == 3:
        dataframe=pd.DataFrame({df_attributes[0]:target_list[0],df_attributes[1]:target_list[1],df_attributes[2]:target_list[2]})
    if len(df_attributes) != len(target_list):
        print('target_list length is not equal to target_list length')
        return dataframe
    if len(df_attributes)<1 or len(df_attributes)>3:
        print('too many/too less attributes, pls rewrite the function by yourself')
    return dataframe

#将dataframe写入csv
#csv_name:"products_urls.csv"
def kangkang_write_csv(csv_name,dataframe):
    dataframe.to_csv(csv_name,index=False,sep=',')
  
'''可视化工具箱'''
#横向plot柱状图
def plot_bar(X,Y,X_label,Y_label,title,color):
    x=range(len(Y))
    plt.barh(x,Y, height=0.7, alpha=0.8, color=color)
    plt.xlabel(str(Y_label))
    plt.ylabel(str(X_label))
    plt.title(str(title))
    plt.yticks([index + 0.2 for index in x], X)
    for x, y in enumerate(Y):
        plt.text(y + 0.2, x - 0.1, '%s' % round(y,2))
    plt.show()

'''bigquery工具箱'''
#申请权限进入IBG数据库
def kangkang_apply_for_query():
    credentials, your_project_id = google.auth.default(scopes=["https://console.cloud.google.com/bigquery?project=ibg-data&refresh=1&_ga=2.26329012.-885081645.1553697324&_gac=1.48956372.1553697398.CjwKCAjwvuzkBRAhEiwA9E3FUm99SDlI5ulzXqDrkiZTFAN8f1W3SQOxiapECsStEySDGqfwrQn8ChoCmgkQAvD_BwE&pli=1&authuser=1"])
    # Make clients.
    bqclient = bigquery.Client(credentials=credentials,project=your_project_id)
    bqstorageclient = bigquery_storage_v1beta1.BigQueryStorageClient(credentials=credentials)
 
#将需要query的内容(query='SELECT * FROM `ibg-data.historical_stage_prod.ext_akeneo_product`')转换成data frame,和list
def kangkang_read_big_query(query):
    start_time=time.time()
    df=pd.read_gbq(query,dialect='standard')
    end_time=time.time()
    cost_time=end_time-start_time
    print('query time cost: ',cost_time,' s')
    mat=df.values.tolist()
    return df,mat

'''正则化工具箱'''
#将句子按输入的符号分开，并以list的形式返还
#symbol: ',' words:'haha,wawa,gaga'
def check_symbol(symbol,words):
    content=[]
    a=re.findall(str("(.*)")+symbol+str("(.*)"),words)[0]
    for i in range(len(a)):
        if symbol in a[i]:
            b=check_symbol(symbol,a[i])
            for j in range(len(b)):
                content.append(b[j])
        else:
            content.append(a[i])
    return content

#将句子全部转换成小写
#word: 'Haha'
def change_to_little(word):
    word=re.sub(r"A","a",word)
    word=re.sub(r"B","b",word)
    word=re.sub(r"C","c",word)
    word=re.sub(r"D","d",word)
    word=re.sub(r"E","e",word)
    word=re.sub(r"F","f",word)
    word=re.sub(r"G","g",word)
    word=re.sub(r"H","h",word)
    word=re.sub(r"I","i",word)
    word=re.sub(r"J","j",word)
    word=re.sub(r"K","k",word)
    word=re.sub(r"L","l",word)
    word=re.sub(r"M","m",word)
    word=re.sub(r"N","n",word)
    word=re.sub(r"O","o",word)
    word=re.sub(r"P","p",word)
    word=re.sub(r"Q","q",word)
    word=re.sub(r"R","r",word)
    word=re.sub(r"S","s",word)
    word=re.sub(r"T","t",word)
    word=re.sub(r"U","u",word)
    word=re.sub(r"V","v",word)
    word=re.sub(r"W","w",word)
    word=re.sub(r"X","x",word)
    word=re.sub(r"Y","y",word)
    word=re.sub(r"Z","z",word)
    return word

#将一个颜色编码转换成一个颜色名称
def regular_color(color):
    if color =='b':
        color='black'
            
    if color =='g':
        color='gray'

    if color == 'lG':
        color='light green'

    if color == 'sg':
        color='silver gray'

    if color == 'o':
        color='orange'

    if color == 'BR':
        color='brown'

    if color == 'skB':
        color='sky blue'
        
    if color == 'dw':
        color='dark white'

    if color == 'dB':
        color='dark blue'

    if color == 'wr':
        color='wine red'

    if color == 'dG':
        color='dark green'

    if color == 'cham':
        color='champagn'

    if color == 'eB':
        color='electric blue'

    if color == 'er':
        color='electric red'

    if color == 'gBR':
        color='gray brown'

    if color == 'eo':
        color='electric orange'

    if color == 'y':
        color='yellow'

    if color == 'r':
        color='red'

    if color == 'go':
        color='gray orange'

    if color == 'dg':
        color='dark gray'

    if color == 'lB':
        color='light blue'

    if color == 'lg':
        color='light gray'

    if color == 'y/b':
        color='black and yellow'

    if color == 'dBR':
        color='dark brown'

    if color == 'G':
        color='green'

    if color == 'eG':
        color='electric green'

    if color == 'b/B':
        color='black and blue'

    if color == 'b/w':
        color='black and white'

    if color == 'b/r':
        color='black and red'

    if color == 'b/G':
        color='black and green'
 
    if color == 'dr':
        color='dark red'

    if color == 'b/lB':
        color='black and light blue'

    if color == 'b/cham':
        color='black and champagn'

    if color == 'b/eB':
        color='black and electric blue'

    if color == 'b/o':
        color='black and orange'

    if color == 'empty':
        color='!'

    if color == 'seaB':
        color='sea blue'

    if color == 'lBR':
        color='light brown'

    if color == 'yg':
        color='yellow gray'

    if color == 'sg/B':
        color='silver gray and blue'

    if color == 'b/sg':
        color='black and silver gray'

    if color == 'g/o':
        color='gray and orange'

    if color == 'b/eG':
        color='black and electric green'

    if color == 'g/er':
        color='gray and electric red'

    if color == 'b/er':
        color='black and electric red'

    if color == 'b/dg':
        color='black and dark gray'

    if color == 'b/seaB':
        color='black and sea blue'

    if color == 'b/g':
        color='black and gray'

    if color == 'b/skB':
        color='black and sky blue'

    if color == 'g/w':
        color='gray and white'
        
    if color == 'B':
        color='blue'
    
    if color == 'w':
        color='white' 
        
    if color == 'w/seaB':
        color='white and sea blue' 
        
    if color == 'g/seaB':
        color='gray and sea blue' 
        
    if color == 'b/y':
        color='black and yellow' 
    
    if color == 'g/r':
        color='gray and red' 
    return color

#将颜色编码list转化成颜色全称list
def regular_colors(colors):
    for i in range(len(colors)):
        colors[i][0]=regular_color(colors[i][0])
    return colors

'''elektrischefietsen工具箱'''
#爬取elektrischefietsen的所有自行车的名称和url，并保存为csv文件
def load_products_urls():
    all_products_urls=[]
    page_index=1
    url_origin='https://www.elektrischefietsen.com/p/'
    while(1):
        if page_index == 1:
            try:
                page=requests.Session().get(url_origin)
                print(page_index)
            except:
                print('no more pages')
                break
        else:
            try:
                url=url_origin+str('page/')+str(page_index)+'/'
                page=requests.Session().get(url)
                print(page_index)
            except:
                print('no more pages')
                break
        page_index+=1
        soup=bs(page.text,'html.parser')
        try:
            sub_urls=soup.find_all(name='div',attrs={"class":"row equal"})[0].find_all('a')
        except:
            print('no more pages')
            break
        for product in sub_urls:
            sub_url=product.get('href')
            sub_page=requests.Session().get(sub_url)
            sub_soup=bs(sub_page.text,'html.parser')
            page_title=str(sub_soup.title.string)
            if page_title == 'Vind de beste e-bike voor jou! | ElektrischeFietsen.com':
                continue
            else:
                bike_name=re.sub(r' \| Elektrischefietsen.com','',page_title)
            all_products_urls.append([sub_url,bike_name])
            #写入csv
            products_urls=[]
            products_names=[]
            for i in range(len(all_products_urls)):
                products_urls.append(all_products_urls[i][0])
                products_names.append(all_products_urls[i][1])
            dataframe=pd.DataFrame({'products_names':products_names,'products_urls':products_urls})
            dataframe.to_csv("products_urls.csv",index=False,sep=',')















