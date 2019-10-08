import re
import matplotlib.image as mpimg # mpimg 用于读取图片
from bs4 import BeautifulSoup as bs
import requests
import urllib
import pandas as pd
import csv
import os

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

def regular(word):
    word=re.sub(r"1. ","",word)
    word=re.sub(r"1.","",word)
    word=re.sub(r"2.",",",word)
    word=re.sub(r"3.",",",word)
    word=re.sub(r"3.",",",word)
    word=re.sub(r" & ",",",word)
    word=re.sub(r" / ",",",word)
    word=re.sub(r"/",",",word)
    word=re.sub(r" , ",",",word)
    word=re.sub(r"/ ",",",word)
    word=re.sub(r" - ",",",word)
    word=re.sub(r"- ",",",word)
    word=re.sub(r"-",",",word)
    word=re.sub(r",",",",word)
    word=re.sub(r" en ",",",word)
    word=re.sub(r"\+",",",word)
    word=re.sub(r"  "," ",word)
    word=re.sub(r" ",",",word)
    word=re.sub(r"\|",",",word)
    unuse=0
    try:
        word=re.findall(r"(.*)=",word)[0]
    except: 
        unuse+=1
    try:
        word=re.findall(r"(.*)\(",word)[0]
    except:  
        unuse+=1
    
    word=change_to_little(word)
    return word


'''
先跑图片验证一下颜色的准确性，然后就可以开始分析流行性了
然后鞍的颜色和把手颜色再打一次标签
'''

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
        
    if color == 'b/m':
        color='black and metal'
        
    if color == 'lo':
        color='light orange'
        
    if color == 'ly':
        color='light yellow'
        
    if color == 'b/lo':
        color='black and light orange'
        
    if color == 'hdxx':
        color='male double thin thin'
        
    if color == 'hsxx':
        color='male single thin thin'
        
    if color == 'dsx':
        color='female single thin'
        
    if color == 'hsvx':
        color='male single very thin'
        
    if color == 'dsc':
        color='female single thick'
        
    if color == 'mdxc':
        color='mountain double thin thick'
        
    if color == 'ddxc':
        color='female double thin thick'
        
    if color == 'f':
        color='fold'
        
    if color == 'hsxc':
        color='male single thin thick'
        
    if color == 'ddxx':
        color='female double thin thin'
        
    if color == 'hscc':
        color='male single thick thick'
        
    if color == 'st':
        color='strange'
        
    if color == 'ddvx':
        color='female double very thin'
        
    if color == 'dou':
        color='double seats'
        
    if color == 'bas':
        color='basket'
        
    if color == 'sp':
        color='special'
    
    return color

def regular_colors(colors):
    for i in range(len(colors)):
        colors[i][0]=regular_color(colors[i][0])
    return colors


def creat_pics_folder(colors,root_folder_name):
    for i in range(len(colors)):
        folder_name=colors[i][0]
        file = "H:\\Desktop\\bike optimization model\\"+str(root_folder_name)+"\\"+folder_name
        os.makedirs(file)   
   
#e_bike_comp是组件记录，root_folder_name是存储的文件夹名，com_column是组件记录里的制定列
def load_pics_to_folders(e_bike_comp,root_folder_name,com_column):
    pics_counter=0
    page_index=0
    url_origin='https://www.elektrischefietsen.com/p/'
    while(1): 
        page_index+=1
        if page_index > 94:
            print('no more pages')
            break
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
        soup=bs(page.text,'html.parser')
        pics=soup.find_all(name='img',attrs={"class":"img-responsive mh-100"})
        names=[]
        sub_urls=soup.find_all(name='div',attrs={"class":"row equal"})[0].find_all('a')
        for i in range(len(sub_urls)):
            names.append(re.findall(r"com[/]p(.*)",sub_urls[i].get('href'))[0])
        for i in range(len(pics)):
            pics_counter+=1
            continue_trigger=1
            for j in range(len(e_bike_comp)):
                if names[i] == e_bike_comp[j][3]:
                    pic_color=regular_color(e_bike_comp[j][com_column])
                    break
                if j == len(e_bike_comp)-1:
                    print("didn't find the name")
                    continue_trigger=0
            if continue_trigger == 0:   
                continue
            pic_src=pics[i].get('src')
            try:
                urllib.request.urlretrieve(pic_src,'H:\\Desktop\\bike optimization model\\'+str(root_folder_name)+'\\'+str(pic_color)+'\\'+str(re.sub(r'[/]','',names[i]))+'.jpg')
            except:
                print(names[i])

            



