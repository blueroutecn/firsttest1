#!/usr/bin/env python
#coding=utf-8


__author__ = 'Haizheng'
__date__ = ''

from conf.config import ver

print(ver)


import pandas as pd
import numpy as np

mydata=pd.DataFrame({
"Name":["苹果","谷歌","脸书","亚马逊","腾讯"],
"Conpany":["Apple","Google","Facebook","Amozon","Tencent"],
"Sale2013":[5000,3500,2300,2100,3100],
"Sale2014":[5050,3800,2900,2500,3300],
"Sale2015":[5050,3800,2900,2500,3300],
"Sale2016":[5050,3800,2900,2500,3300]
       })



print(mydata.head())

mydata1=mydata.melt(
id_vars=["Name","Conpany"],   #要保留的主字段
var_name="Year",                     #拉长的分类变量
value_name="Sale"                  #拉长的度量值名称
        )

# print(mydata1.head(20))




print(mydata1.pivot_table(
index=["Name","Conpany"],    #行索引（可以使多个类别变量）
columns=["Year"],                   #列索引（可以使多个类别变量）
values=["Sale"]                       #值（一般是度量指标）
     ))



