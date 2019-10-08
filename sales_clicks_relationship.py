import kangkang_tools_box as kk
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

'''
申请毕业
还可以加上GA、behavior、events、top events、event label、secondary dimension：page去看每个自行车跳出网页去购买的比例
还有全部的搜索空间表现出来
还有就是怎么验证我的预测（时间段选取前面一段做预测，然后看后面的点击率是否符合）
Label可否变为点击率+销售额？考虑一下

上午完成：
写出超参数优化的算法开始跑
'''


#洗GA数据
def load_elektrischefietsen_hyperlink_data():
    hyperlink_mat=[]
    file_index=1
    while(1):
        file_name='./hyperlink/'+str(file_index)+'.csv'
        try:
            current_file= pd.read_csv(file_name,sep=',', skiprows=6)
            current_file=current_file.values
            current_file=current_file.tolist()
        except:
            break
        for i in range(len(current_file)):
            hyperlink_mat.append(current_file[i])
        file_index+=1
    return hyperlink_mat

def get_hyperlink_mat(e_bike_comp):
    hyperlink_mat=load_elektrischefietsen_hyperlink_data()
    new_hyperlink_mat=[]
    for i in range(len(hyperlink_mat)):
        try:
            if 'fietsenwinkel' in hyperlink_mat[i][0]:
                new_hyperlink_mat.append(hyperlink_mat[i])
        except:
            useless=1
    hyperlink_mat=new_hyperlink_mat
    new_hyperlink_mat=[]
    for i in range(len(hyperlink_mat)):
        for j in range(len(e_bike_comp)):
            try:
                if e_bike_comp[j][3] in hyperlink_mat[i][1] and 'fietsenwinkel' in hyperlink_mat[i][0]:
                    new_hyperlink_mat.append([re.sub(r'https://www.fietsenwinkel.nl','',hyperlink_mat[i][0]),re.sub(r'/p/','/',hyperlink_mat[i][1])])
                    break
            except:
                useless=1
    return new_hyperlink_mat


def query_for_new_date():
    query='''SELECT
      *
    FROM (
      SELECT
        p.V2productName AS Product,
        p.productSKU AS MagentoSKU,
        hits.page.pagePath AS Page,
        SUM(totals.visits) AS visits
      FROM
        `ibg-data.190250099.ga_sessions_*`,
        UNNEST(hits) AS hits,
        UNNEST(hits.product) AS p
      WHERE
        _table_suffix BETWEEN '20190325'
        AND '20190812'
        AND hits.eCommerceAction.action_type = '2'
      GROUP BY
        p.V2productName,
        p.productSKU,
        Page )
    LEFT JOIN (
      SELECT
        SUM(Quantity) AS Quantity,
        SUM(CM3_Euro) AS CM3_Euro,
        REPLACE(ParentSKU, '_', '-') AS MagentoSKU
    
      FROM
        `ibg-data.data_mart.SalesFacts`
      WHERE
        ORderTimestamp BETWEEN '2019-03-25'
        AND '2019-07-10'
        AND ProductCategory1 = 'Bikes'
        AND Channel = 'Online'
      GROUP BY
        MagentoSKU)
    USING
      (MagentoSKU)
    WHERE
      Quantity IS NOT NULL
    '''
    kk.kangkang_apply_for_query()
    #将需要query的内容(query='SELECT * FROM `ibg-data.historical_stage_prod.ext_akeneo_product`')转换成data frame,和list
    sale_df,sale_mat=kk.kangkang_read_big_query(query)
    kk.kangkang_write_csv('sales_fact.csv',sale_df)
    return sale_df,sale_mat

def trace_sales_fact(hyperlink_mat,sales_fact):
    for i in range(len(hyperlink_mat)):
        for j in range(len(sales_fact)):
            if re.sub(r'/','',hyperlink_mat[i][0]) in re.sub(r'/','',sales_fact[j][2]) or re.sub(r'/','',sales_fact[j][2]) in re.sub(r'/','',hyperlink_mat[i][0]):
                if pd.isnull(sales_fact[j][3]) == True:
                    sales_fact[j][3]=0
                if pd.isnull(sales_fact[j][4]) == True:
                    sales_fact[j][4]=0
                if pd.isnull(sales_fact[j][5]) == True:
                    sales_fact[j][5]=0
                hyperlink_mat[i].append([sales_fact[j][3],sales_fact[j][4],sales_fact[j][5]])
    return hyperlink_mat

def merge_sales(hyperlink_mat):
    for i in range(len(hyperlink_mat)):
        if len(hyperlink_mat[i]) > 3:
            merger=[]
            for j in range(len(hyperlink_mat[i])-2):
                merger.append(hyperlink_mat[i][2+j])
            merger=sorted(merger, key=lambda x:x[0],reverse = True)
            hyperlink_mat[i]=[hyperlink_mat[i][0],hyperlink_mat[i][1],merger[0]]
    return hyperlink_mat

#hyperlink_mat上加有效点击率
def plus_valid_click_rate(hyperlink_mat):
    for i in range(len(hyperlink_mat)):
        for j in range(len(labels_source)):
            if labels_source[j][0] == re.sub(r'/','',hyperlink_mat[i][1]):
                if len(hyperlink_mat[i]) == 2:
                    hyperlink_mat[i].append([0,0,0,labels_source[j][1]])
                    break
                if len(hyperlink_mat[i]) == 3:
                    hyperlink_mat[i][2].append(labels_source[j][1])
                    break
    new_hyperlink_mat=[]
    for i in range(len(hyperlink_mat)):
        try:
            if len(hyperlink_mat[i][2]) == 4 :#and hyperlink_mat[i][2][0] != 0
                new_hyperlink_mat.append(hyperlink_mat[i])
        except:
            useless=0
    hyperlink_mat=new_hyperlink_mat
    return hyperlink_mat

#盒状图
def box_plot(Y_click,Y_sale):
    box_size=60
    sale_X=list(range(int(len(Y_click)/box_size)+1))
    click_box=[]
    sale_box=[]
    boxes_counter=0
    ele_counter=0
    for i in range(len(Y_click)):
        if i%box_size == 0 or i ==0:
            click_box.append([])
            sale_box.append([])
            for j in range(box_size):
                try:
                    click_box[boxes_counter].append(Y_click[ele_counter])
                    sale_box[boxes_counter].append(Y_sale[ele_counter])
                    ele_counter+=1
                except:
                    useless=1
            boxes_counter+=1
    click_box=np.array(click_box)
    sale_box=np.array(sale_box)
    
    color = dict(boxes='DarkGreen',whiskers='DarkOrange',medians='DarkBlue',caps='Gray')
    plt.figure()
    plt.boxplot(sale_box,labels=sale_X,sym = "b",boxprops = {'color':'blue'})
    plt.boxplot(click_box,labels=sale_X,sym = "r",boxprops = {'color':'red'})
    plt.title('box_plot relevance between Sales volume and valid clicks')
    plt.xlabel(str(box_size)+' bikes per box') 
    plt.ylabel('rate after standardization')       #x轴的标签
    plt.show()
    
#散点图
def scatter_plot(Y_sale,Y_click):
    X=list(range(int(len(Y_click))))
    plt.figure()
    plt.scatter(X,Y_sale,marker='x',label='Sales volume')
    plt.scatter(X,Y_click,marker='v',label='valid click',alpha=0.6)
    plt.title('scatter_plot relevance between Sales volume and valid clicks',alpha=0.6)
    plt.xlabel('bikes')       #x轴的标签
    plt.ylabel('rate after standardization')
    plt.legend()
    plt.show()
    
#比例图
def plot_rate(Y_click,Y_sale):
    X=list(range(int(len(Y_click))))
    Y_rate=[]
    up_line=[]
    bottom_line=[]
    standard_set=[]
    for i in range(len(Y_click)):
        Y_rate.append(Y_sale[i]-Y_click[i])
        standard_set.append(abs(Y_sale[i]-Y_click[i]+1))
    standard_set=np.array(standard_set)
    standard=np.mean(standard_set)
    for i in range(len(Y_click)):
        up_line.append(standard)
        bottom_line.append(standard*(-1))
    plt.figure()
    plt.scatter(X,Y_rate,marker='x',label='gap size',color='g')
    plt.plot(X,up_line,label='outlier_up_boundary',color='r')
    plt.plot(X,bottom_line,label='outlier_bottom_boundary',color='r')
    plt.title('scatter_plot gap size between Sales volume and valid clicks')
    plt.xlabel('bikes')  
    plt.ylabel('gap size')  
    plt.legend()
    plt.show()
    return Y_rate,standard

#找出奇点
def find_outliers(Y_rate,hyperlink_mat,standard):
    outlier_up=[]
    outlier_bottom=[]
    for i in range(len(Y_rate)):
        if Y_rate[i] > standard:
            outlier_up.append([hyperlink_mat[i][0],['revenue-clicks',Y_rate[i]],['clicks',hyperlink_mat[i][2][click_source]],['revenue',hyperlink_mat[i][2][sale_source]]])
        if Y_rate[i] < (-1)*standard:
            outlier_bottom.append([hyperlink_mat[i][0],['revenue-clicks',Y_rate[i]],['clicks',hyperlink_mat[i][2][click_source]],['revenue',hyperlink_mat[i][2][sale_source]]])
    outlier_up=sorted(outlier_up, key=lambda x:x[1][1],reverse = True)
    outlier_bottom=sorted(outlier_bottom, key=lambda x:x[1][1],reverse = False)
    return outlier_up,outlier_bottom

#找出目前有组件记录的自行车
def find_number_of_bikes_exist_in_both_web(e_bike_comp,hyperlink_mat):
    number_of_bikes_exist_in_both_web=0
    for i in range(len(e_bike_comp)):
        for j in range(len(hyperlink_mat)):
            if hyperlink_mat[j][1] in e_bike_comp[i][3]:
                number_of_bikes_exist_in_both_web+=1
                break
    return number_of_bikes_exist_in_both_web

def abs_Y_sale_and_mean(Y_sale):
    mean=0
    for i in range(len(Y_sale)):
        mean+=abs(Y_sale[i])
    mean=mean/len(Y_sale)
    return mean
###############################################################################
e_bike_comp = pd.read_csv('e_bike_excel_copy.csv',sep=';')
e_bike_comp=e_bike_comp.values
e_bike_comp=e_bike_comp.tolist()
hyperlink_mat=get_hyperlink_mat(e_bike_comp)
sales_fact = pd.read_csv('sales_fact.csv',sep=',')
sales_fact=sales_fact.values
sales_fact=sales_fact.tolist()
hyperlink_mat=trace_sales_fact(hyperlink_mat,sales_fact)
hyperlink_mat=merge_sales(hyperlink_mat)
hyperlink_mat=plus_valid_click_rate(hyperlink_mat)

click_source=3
sale_source=1

#可视化
Y_click=[]
Y_sale=[]
Y_click_abs=[]
Y_sale_abs=[]
for i in range(len(hyperlink_mat)):
    if pd.isnull(hyperlink_mat[i][2][click_source]) == True:
        Y_click.append(0)
        Y_click_abs.append(0)
    if pd.isnull(hyperlink_mat[i][2][click_source]) != True:
        Y_click.append(hyperlink_mat[i][2][click_source])
        Y_click_abs.append(abs(hyperlink_mat[i][2][click_source]))
    if pd.isnull(hyperlink_mat[i][2][sale_source]) == True:
        Y_sale.append(0)
        Y_sale_abs.append(0)
    if pd.isnull(hyperlink_mat[i][2][sale_source]) != True:
        Y_sale.append(hyperlink_mat[i][2][sale_source])
        Y_sale_abs.append(abs(hyperlink_mat[i][2][sale_source]))
mix=[]

diff=np.mean(Y_sale_abs)/np.mean(Y_click_abs)
for i in range(len(Y_sale)):
    Y_sale[i]=(Y_sale[i]-np.mean(Y_sale))/np.std(Y_sale,ddof=1)
    Y_click[i]=(Y_click[i]-np.mean(Y_click))/np.std(Y_click,ddof=1)
    mix.append([Y_click[i],Y_sale[i]])

mix=sorted(mix, key=lambda x:x[0],reverse = True)
mix=np.array(mix)
Y_click=mix[:,0]
Y_sale=mix[:,1] 
box_plot(Y_click,Y_sale)
scatter_plot(Y_sale,Y_click)
Y_rate,standard=plot_rate(Y_click,Y_sale)
outlier_up,outlier_bottom=find_outliers(Y_rate,hyperlink_mat,standard)
number_of_bikes_exist_in_both_web=find_number_of_bikes_exist_in_both_web(e_bike_comp,hyperlink_mat)

coef=np.corrcoef([Y_sale,Y_click])
#正太分布图
'''
plt.figure()
mu, sigma , num_bins = 0, 1, int(len(Y_click))
n, bins, patches =plt.hist(Y_rate, bins=num_bins, normed=True)
y = mlab.normpdf(bins, mu, sigma)
#plt.plot(bins, y, 'r--')
plt.title('Normal distribution of revenue-clicks')
plt.show()
'''




        

#valid clicks per day把bunce rate改成跳转率怎么样
#用这600量的revenue-click作为标签来训练怎么样








