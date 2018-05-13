#!/usr/bin/env python
# coding=utf-8


__author__ = 'Haizheng'
__date__ = ''

import pandas as pd
import numpy as np


def lab1():
    # 按列生成数据
    df = pd.DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'a', 'b'], 'data1': range(7)})
    df1 = pd.DataFrame({
        'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
        'year': [2001, 2001, 2002, 2001, 2002],
        'pop': [1.5, 1.7, 3.6, 2.4, 2.9]
    })
    print(df1)
    print(df1.query("year =='2001'"))
    # 数据汇总
    # print(df1.groupby(['year', 'state'])['pop'].sum(axis=1, skipna=False))
    # print(df1.groupby(['year', 'state'])['pop'].agg(
    #     {"sum":"sum","count":"count"}).unstack('state').head(2))

    # print(df1.groupby(['year', 'state'])['pop'].aggregate({"pers":["sum", "count"],"totel":"max"}).unstack("state"))

    print(df1.pivot_table('pop', index=['year'], columns=['state'], aggfunc="sum"))

    df = pd.DataFrame({'A': [1, 1, 2, 2],
                       'B': [1, 2, 3, 4],
                       'C': np.random.randn(4)})
    print(df)
    print(df.groupby('A').agg({'B': ['min', 'max'], 'C': 'sum'}))

    df = pd.DataFrame(np.random.randn(10, 2), columns=list('ab'))
    print(df)
    print(df.eval('c = a+b'))

    df = pd.DataFrame(np.random.randn(5, 2), columns=list('ab'))
    # 本地变量
    newcol = np.random.randn(len(df))
    print(df)
    print(df.eval('c = b + @newcol'))

    a, b = 1, 2
    print(pd.eval('a + b'))


def re_lab():
    import re

    """
    1、搜索包含@或则at
    2、包含dot或.
    3、de结尾
    4、文本中没有空格
    
    正则复习：
    .    非\n
    [...]  范围[a-c] 去反^ [^abc], 以开头^abc
    \\b
    \\d  数字
    \\s  空白
    \\w  单词
    
    数量次：
    * *?  0-unlimit
    . .?  1-unlimit
    ? ??  0-1
    {m} {m}?  m-unpimit
    
    边界匹配
    ^
    $
    \A
    \Z
    \\b   匹配一个字边界，即字与空格间的位置。
    \\B   非字边界匹配。
    
    分组
    特殊构造（不做分组）消除匹配缓存
    ?:
    ?= 正向 
    ?! 反向 
    
    捕获的都要带上次数
    """

    mail_li = ['lokesh.g@uni-passau.de',
               'lokesh dot gat uni-passau dot de',
               'lokesg.g(at)uni-passau.de',
               'lokesh.g(at) uni-passau.de',
               'onefullword@gw.uni-passau.de',
               'lokesh(dot) g(at) uni-passau (dot) de',
               '@laskdjak']

    regex = r"^(?=.*?\b(?:@|\(at\))\b).*(?=\b(?:\.|\(dot\))de$).*$"

    for i in mail_li:
        if (re.findall(regex, i)):
            print(i)


def async_lab():
    """
    async is used to declare a function as a coroutine,
    much like what the @asyncio.coroutine decorator does.
    It can be applied to the function by putting it at the front of the definition:
    async:通常用来定义一个函数为协程，类是@asyncio.coroutine的装饰器
    实际调用这个方法，我们使用await,代替以前的yield from
    :return:
    """
    def ping_server(ip):
        pass

    async def ping_server(ip):
        return await ping_server('192.168.1.1')
        # ping code here...


    # for i in aiter():
    #     print(i)

def subclass_lab():
    """
    定制类的创建使用新协议进行了简化

    描述符是一个具有绑定行为的对象属性
    它的优先级高会改变一个属性的基本的获取、设置和删除方式

    """
    class PluginBase:
        subclass = []

        def __init_subclass__(cls, **kwargs):
            super().__init_subclass__(**kwargs)
            cls.subclass.append(cls)

    class Plugin1(PluginBase):
        pass

    class Plugin2(PluginBase):
        pass

    print(PluginBase.subclass)

    class Integer(object):
        def __get__(self, instance, owner):
            return instance.__dict__[self.name]

        def __set__(self, instance, value):
            if value < 0:
                raise ValueError('Negative value not allowed')
            instance.__dict__[self.name] = value

        def __set_name__(self, owner, name):
            self.name = name

    class Movie(object):
        score = amount = Integer()




if __name__ == '__main__':
    # lab1()
    # re_lab()
    subclass_lab()
