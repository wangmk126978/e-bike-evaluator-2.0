import requests
import re
from bs4 import BeautifulSoup as bs
import pandas as pd
import numpy as np


e_bike_comp = pd.read_csv('e_bike_excel_copy.csv',sep=';')
e_bike_comp=e_bike_comp.values
e_bike_comp=e_bike_comp.tolist()

#这个函数不光可以用来编辑品牌哟
#创建空的品牌列表
bike_brand=[]
for i in range(len(e_bike_comp)):
    bike_brand.append([e_bike_comp[i][3]])
#开始填充品牌列表
brand_affection=[['traploos'],['1'],['2'],['3'],['4'],['5'],['6'],['7'],['8'],['9'],['10'],['11'],['12'],['14'],['17'],['20'],['22'],['24'],['27'],['30']]
url_origin='https://www.elektrischefietsen.com/p/page/'
for i in range(len(brand_affection)):
    page_index=1
    while(1):
        url=url_origin+str(page_index)+'/?aantal_versnellingen[]='+brand_affection[i][0]
        page=requests.Session().get(url)
        soup=bs(page.text,'html.parser')
        try:
            sub_urls=soup.find_all(name='div',attrs={"class":"row equal"})[0].find_all('a')
            for j in range(len(sub_urls)):
                bike_of_this_brand=sub_urls[j].get('href')
                for z in range(len(bike_brand)):
                    if bike_brand[z][0] in bike_of_this_brand:
                        bike_brand[z].append(brand_affection[i][0])
                        break
                    if z == len(bike_brand)-1:
                        print('the bike is new')
        except:
            print('no more pages')
            break
        page_index+=1

brands=[]
a=[]
for i in range(len(bike_brand)):
    if len(bike_brand[i]) == 2:
        brands.append(re.sub(r'-elektrische-fietsen','',bike_brand[i][1]))
    if len(bike_brand[i]) > 2:
        avg_num=0
        for j in range(len(bike_brand[i])):
            if j == 0:
                continue
            avg_num+=int(bike_brand[i][j])
        avg_num=int(avg_num/(len(bike_brand[i])-1))
        brands.append(re.sub(r'-elektrische-fietsen','',str(avg_num)))
    if len(bike_brand[i])<2:
        brands.append(re.sub(r'-elektrische-fietsen','','no'))
        a.append(bike_brand[i])
        
        
        
