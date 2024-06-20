#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 08:51:53 2024

@author: gl.novikov
"""

import os
os.chdir("/Users/gl.novikov/Work")

# Импорт необходимых библиотек
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pandas.api.types import CategoricalDtype
#Создание ДатаФрейма
pth = '/Users/gl.novikov/Work/Data/googleplaystore.csv' # Путь относительно рабочего каталога
GOOGLE = pd.read_csv(pth)

#Удаление строк с пропущенными значениями
GOOGLE = GOOGLE.dropna()
GOOGLE = GOOGLE.reset_index()
del GOOGLE['index']
GOOGLE.drop('Genres',axis = 1, inplace =True)
#Присвоение перменным Category, Type, Сontent Rating, Genres типа категория
GOOGLE['Category'] = GOOGLE['Category'].astype("category")
GOOGLE['Type'] = GOOGLE['Type'].astype("category")
GOOGLE['Content Rating'] = GOOGLE['Content Rating'].astype("category")




#Обработка данных переменной Price: удаление неудачных случаев, переформатирование данных ввиде $4.99 -> 4.99 и преобразование к типу float
GOOGLE.drop(GOOGLE.loc[GOOGLE['Price'] == 'Everyone'].index, inplace=True)
GOOGLE = GOOGLE.reset_index()
del GOOGLE['index']
for i in range(len(GOOGLE)):
    if GOOGLE.loc[i,'Price'] != '0':
        GOOGLE.loc[i,'Price'] = GOOGLE.loc[i,'Price'][1:]
GOOGLE['Price'] = GOOGLE['Price'].astype("float64")

#Присвоение переменной Reviews типа категории
GOOGLE['Reviews'] = GOOGLE['Reviews'].astype("int64")

#Обработка данных переменной Size: удаление неудачных случаев, переформатирование данных ввиде 1.4M >1.4, преобразование к типу float
GOOGLE.drop(GOOGLE.loc[GOOGLE['Size'] == 'Varies with device'].index, inplace=True)
GOOGLE = GOOGLE.reset_index()
del GOOGLE['index']
for i in range(len(GOOGLE)):
    if GOOGLE.loc[i,'Size'][-1] == 'k':
        GOOGLE.loc[i,'Size'] = GOOGLE.loc[i,'Size'][:-1]
        GOOGLE.loc[i,'Size'] = round(float(GOOGLE.loc[i,'Size'])/1024,3)
    elif GOOGLE.loc[i,'Size'][-1] == 'M':
        GOOGLE.loc[i,'Size'] = float(GOOGLE.loc[i,'Size'][:-1])
GOOGLE['Size'] = GOOGLE['Size'].astype("float64")
#Обработка данных переменной Last Updated: преобразование данных ввиде 13 January,2018 -> 2018, присвоение типа категория
for i in range(len(GOOGLE)):
    tmp = GOOGLE.loc[i,'Last Updated'].split()
    GOOGLE.loc[i,'Last Updated'] = (tmp[0]+tmp[2])[-4:]   
GOOGLE['Last Updated'] = GOOGLE['Last Updated'].astype("category")

#Обратботка данных переменной Installs: сужение алфавита до 10 путем логичного преобразования данных, преобразование переменной к периодическому типу
for i in range(len(GOOGLE)):
    if GOOGLE.loc[i,'Installs'] == '1+':
        GOOGLE.loc[i,'Installs'] = '0+'
    elif GOOGLE.loc[i,'Installs'] == '10+':
        GOOGLE.loc[i,'Installs'] = '5+'
    elif GOOGLE.loc[i,'Installs'] == '100+':
        GOOGLE.loc[i,'Installs'] = '50+'
    elif GOOGLE.loc[i,'Installs'] == '1,000+':
        GOOGLE.loc[i,'Installs'] = '500+'
    elif GOOGLE.loc[i,'Installs'] == '10000+':
        GOOGLE.loc[i,'Installs'] = '5000+'
    elif GOOGLE.loc[i,'Installs'] == '1,000,000,000+':
        GOOGLE.loc[i,'Installs'] = '500,000,000+'        
    elif GOOGLE.loc[i,'Installs'] == '100,000+':
        GOOGLE.loc[i,'Installs'] = '50,000+'
    elif GOOGLE.loc[i,'Installs'] == '10,000+':
        GOOGLE.loc[i,'Installs'] = '5,000+'   
    elif GOOGLE.loc[i,'Installs'] == '1,000,000+':
        GOOGLE.loc[i,'Installs'] = '500,000+' 
    elif GOOGLE.loc[i,'Installs'] == '10,000,000+':
        GOOGLE.loc[i,'Installs'] = '5,000,000+' 
    elif GOOGLE.loc[i,'Installs'] == '100,000,000+':
        GOOGLE.loc[i,'Installs'] = '500,000,000+' 
list_of_installs  = sorted(list(set(GOOGLE['Installs'])),key=len)
list_of_installs[0],list_of_installs[1]=list_of_installs[1],list_of_installs[0]
installs_type = CategoricalDtype(categories=list_of_installs,ordered = True)
GOOGLE['Installs']= GOOGLE['Installs'].astype(installs_type)


#Обработка данных переменной Android Ver: удаление неудачных случаев, переформативоние данных ввиде v.4.0.2.1 and up -> v.4 and up, сужение алфавита до 10,  преобразование переменной к периодическому типу
GOOGLE.drop(GOOGLE.loc[GOOGLE['Android Ver'] == 'Varies with device'].index, inplace=True)
GOOGLE = GOOGLE.reset_index()
del GOOGLE['index']
for i in range(len(GOOGLE)):
    GOOGLE.loc[i,'Android Ver'] = 'v.'+str(GOOGLE.loc[i,'Android Ver'])[0]+' and up'     
set_of_android_version = list(set(GOOGLE['Android Ver']))
list_of_android_version = sorted(set_of_android_version)
android_type = CategoricalDtype(categories=list_of_android_version,ordered = True)
GOOGLE['Android Ver']= GOOGLE['Android Ver'].astype(android_type)

#Обработка данных переменной Current:удаление неудачных случаев,  переформативоние данных ввиде v.4.0.2-> v.4, сужение алфавита до 10,  преобразование переменной к периодическому типу
GOOGLE.drop(GOOGLE.loc[GOOGLE['Current Ver'] == 'Varies with device'].index, inplace=True)
GOOGLE = GOOGLE.reset_index()
del GOOGLE['index']
for i in range(len(GOOGLE)):
    GOOGLE.loc[i,'Current Ver'] = str(GOOGLE.loc[i,'Current Ver'])[0]

GOOGLE.drop(GOOGLE.loc[GOOGLE['Current Ver'].str.isalpha()].index, inplace=True)
GOOGLE = GOOGLE.reset_index()
del GOOGLE['index']

for i in range(len(GOOGLE)):
    GOOGLE.loc[i,'Current Ver'] = 'v.' + str(GOOGLE.loc[i,'Current Ver'])[0]

set_of_current_version = list(set(GOOGLE['Current Ver']))
list_of_current_version = sorted(set_of_current_version)
android_cur = CategoricalDtype(categories=list_of_current_version,ordered = True)
GOOGLE['Current Ver']= GOOGLE['Current Ver'].astype(android_cur) 

set_of_cats = list(set(GOOGLE['Category']))
for i in range(len(GOOGLE)):
    if GOOGLE.loc[i,'Category'] in ['DATING','LIFESTYLE','COMMUNICATION']:
        GOOGLE.loc[i,'Category'] = 'SOCIAL'
    elif GOOGLE.loc[i,'Category'] in ['COMICS','NEWS_AND_MAGAZINES']:
        GOOGLE.loc[i,'Category'] = 'BOOKS_AND_REFERENCE'
    elif GOOGLE.loc[i,'Category'] in ['PERSONALIZATION','LIBRARIES_AND_DEMO','PHOTOGRAPHY',
                                      'PRODUCTIVITY','MAPS_AND_NAVIGATION','VIDEO_PLAYERS',
                                      'ART_AND_DESIGN','MEDICAL','WEATHER','AUTO_AND_VEHICLES']:
        GOOGLE.loc[i,'Category'] = 'TOOLS'
    elif GOOGLE.loc[i,'Category'] in ['HOUSE_AND_HOME','PARENTING']:
        GOOGLE.loc[i,'Category'] = 'FAMILY'
    elif GOOGLE.loc[i,'Category'] in['BUSINESS']:
        GOOGLE.loc[i,'Category'] = 'FINANCE'
    elif GOOGLE.loc[i,'Category'] in ['TRAVEL_AND_LOCAL','EVENTS']:
        GOOGLE.loc[i,'Category'] = 'ENTERTAINMENT'
    elif GOOGLE.loc[i,'Category'] in ['SPORTS','BEAUTY']:
        GOOGLE.loc[i,'Category'] = 'HEALTH_AND_FITNESS'
    elif GOOGLE.loc[i,'Category'] in ['FOOD_AND_DRINK']:
        GOOGLE.loc[i,'Category'] = 'SHOPPING'
# ********************** Числовой анализ *************************
# Количественный анализ количественных переменных.
from scipy.stats import skew,kurtosis
GOOG = GOOGLE.select_dtypes(include=['float','int'])
# Описательная статистика всех переменных
GOOGLE_STAT = GOOG.describe()
# Медиана для всех переменных
GOOGLE_med = GOOG.median() # Получается pandas.Series
# Межквартильный размах для всех переменных
# Вычисляется по определению
GOOGLE_iqr = GOOG.quantile(q=0.75) - GOOG.quantile(q=0.25) # Получается pandas.Series
# Создаем pandas.DataFrame из новых статистик
W = pd.DataFrame([GOOGLE_iqr], index=['IQR'])
# Объединяем GOOGLR_STAT и W
GOOGLE_STAT = pd.concat([GOOGLE_STAT, W])

GOOG_sk = GOOG.skew()
S = pd.DataFrame([GOOG_sk],index = ['skew'])
GOOGLE_STAT = pd.concat([GOOGLE_STAT, S])

GOOG_kur = GOOG.kurtosis()
K = pd.DataFrame([GOOG_kur],index = ['kurtosis'])
GOOGLE_STAT = pd.concat([GOOGLE_STAT, K])

import scipy as sp
# Outliers
std_p = np.std(GOOGLE['Price'])
mean_p = np.mean(GOOGLE['Price'])
sel_out_p = np.abs(GOOGLE['Price'] - mean_p) > 3*std_p
GOOGLE_OUT_P = GOOGLE.loc[sel_out_p, :]

std_ra = np.std(GOOGLE['Rating'])
mean_ra = np.mean(GOOGLE['Rating'])
sel_out_ra = np.abs(GOOGLE['Rating'] - mean_ra) > 3*std_ra
GOOGLE_OUT_RA = GOOGLE.loc[sel_out_ra, :]

std_re = np.std(GOOGLE['Reviews'])
mean_re = np.mean(GOOGLE['Reviews'])
sel_out_re = np.abs(GOOGLE['Reviews'] - mean_re) > 3*std_re
GOOGLE_OUT_RE = GOOGLE.loc[sel_out_re, :]

std_s = np.std(GOOGLE['Size'])
mean_s = np.mean(GOOGLE['Size'])
sel_out_s = np.abs(GOOGLE['Size'] - mean_s) > 3*std_s
GOOGLE_OUT_S = GOOGLE.loc[sel_out_s, :]


# std = k*MAD, k=1.48... fof N()
mad = sp.stats.median_abs_deviation(GOOGLE['Price'])
tr_mean = sp.stats.trim_mean(proportiontocut=0.1, a = GOOGLE['Price'])
with pd.ExcelWriter('/Users/gl.novikov/Work/Output/GOOGLE_STAT.xlsx', engine="openpyxl", 
                    if_sheet_exists='overlay', mode='a') as wrt:
    GOOGLE_STAT.to_excel(wrt, sheet_name='STATS') 


from scipy.stats import pearsonr
from scipy.stats import spearmanr

# Здесь будут значения оценок коэффициента корреляции Пирсона
C_P = pd.DataFrame([], index=GOOG.columns, columns=GOOG.columns) 
# Здесь будут значения значимости оценок коэффициента корреляции Пирсона
P_P = pd.DataFrame([], index=GOOG.columns, columns=GOOG.columns)
# Здесь будут значения оценок коэффициента корреляции Спирмена
C_S = pd.DataFrame([], index=GOOG.columns, columns=GOOG.columns)
# Здесь будут значения значимости оценок коэффициента корреляции Спирмена
P_S = pd.DataFrame([], index=GOOG.columns, columns=GOOG.columns)
for x in GOOG.columns:
    for y in GOOG.columns:
        C_P.loc[x,y], P_P.loc[x,y] = pearsonr(GOOG[x], GOOG[y])
        C_S.loc[x,y], P_S.loc[x,y] = spearmanr(GOOG[x], GOOG[y])

# Сохраняем текстовый отчет на разные листы Excel файла
with pd.ExcelWriter('/Users/gl.novikov/Work/Output/GOOGLE_STAT.xlsx', engine="openpyxl") as wrt:
# Общая статистика
    GOOGLE_STAT.to_excel(wrt, sheet_name='stat')
# Корреляция Пирсона
    C_P.to_excel(wrt, sheet_name='Pearson')
    dr = C_P.shape[0] + 2
    P_P.to_excel(wrt, startrow=dr, sheet_name='Pearson') # Значимость
# Корреляция Спирмена
    C_S.to_excel(wrt, sheet_name='Spirmen')
    dr = C_S.shape[0] + 2
    P_S.to_excel(wrt, startrow=dr, sheet_name='Spirmen') # Значимость

import math
plt.figure(figsize=(16,6))
plt.subplot(121)
plt.hist(GOOGLE['Rating'], color = 'lightblue', edgecolor = 'black',
         bins = 'sturges')
plt.title('Рейтинг приложения')
plt.xlabel('Рейтинг')
plt.ylabel('Относительная частота')
plt.text(2.5, 2000,'По Стерджессу')
plt.subplot(122)
plt.hist(GOOGLE['Rating'], color = 'lightblue', edgecolor = 'black',
         bins = 'fd')
plt.title('Рейтинг приложения')
plt.xlabel('Рейтинг')
plt.ylabel('Относительная частота')
plt.text(2.5, 700,'По Фрид.-Диак')
plt.savefig('./Graphics/Rating.pdf', format='pdf')



plt.figure(figsize=(16,6))
plt.subplot(121)
GOOGLE_copy = GOOGLE.copy()
for i in range(len(GOOGLE)):
    if GOOGLE_copy.loc[i,'Reviews']>=38367: 
        GOOGLE_copy.loc[i,'Reviews'] = np.nan
plt.hist(GOOGLE_copy['Reviews'], color = 'lightblue', edgecolor = 'black',
         bins = 'sturges')
plt.title('Количество отзывов')
plt.xlabel('Отзывы')
plt.ylabel('Относительная частота')
plt.text(15000, 3000,'По Стерджессу')
plt.subplot(122)
plt.hist(GOOGLE_copy['Reviews'], color = 'lightblue', edgecolor = 'black',
         bins = 'fd')
plt.title('Количество отзывов')
plt.xlabel('Отзывы')
plt.ylabel('Относительная частота')
plt.text(15000, 2250,'По Фрид.-Диак')
plt.savefig('./Graphics/Reviews.pdf', format='pdf')


plt.figure(figsize=(16,6))
plt.subplot(121)
plt.hist(GOOGLE['Size'], color = 'lightblue', edgecolor = 'black',
         bins = 'sturges')
plt.title('Размер приложения')
plt.xlabel('Размер')
plt.ylabel('Относительная частота')
plt.text(40, 2000,'По Стерджессу')
plt.subplot(122)
plt.hist(GOOGLE['Size'], color = 'lightblue', edgecolor = 'black',
         bins = 'fd')
plt.title('Размер приложения')
plt.xlabel('Размер')
plt.ylabel('Относительная частота')
plt.text(40, 1000,'По Фрид.-Диак')
plt.savefig('./Graphics/Size.pdf', format='pdf')


plt.figure(figsize=(16,6))
GOOGLE_copy = GOOGLE.copy()
for i in range(len(GOOGLE)):
    if GOOGLE_copy.loc[i,'Price']==0 or GOOGLE_copy.loc[i,'Price']>79: 
        GOOGLE_copy.loc[i,'Price'] = np.nan
plt.subplot(121)
plt.hist(GOOGLE_copy['Price'], color = 'lightblue', edgecolor = 'black',
         bins = 'sturges')
plt.title('Цена за приложение')
plt.xlabel('Цена')
plt.ylabel('Относительная частота')
plt.text(15, 300,'По Стерджессу')
plt.subplot(122)
plt.hist(GOOGLE_copy['Price'], color = 'lightblue', edgecolor = 'black',
         bins = 'fd')
plt.title('Цена за приложение')
plt.xlabel('Цена')
plt.ylabel('Относительная частота')
plt.text(15, 120,'По Фрид.-Диак')
plt.savefig('./Graphics/Price.pdf', format='pdf')

# Анализ корреляции между количественной целевой переменной
# и качественной объясняющей
# Используем библиотеку scipy
# Критерий Крускала-Уоллиса
from scipy.stats import kruskal
# Создаем подвыборки
sel_1 = GOOGLE['Category']=='SOCIAL'
x_1 = GOOGLE.loc[sel_1, 'Rating']

sel_2 = GOOGLE['Category']=='BOOKS_AND_REFERENCE'
x_2 = GOOGLE.loc[sel_2, 'Rating']

sel_3 = GOOGLE['Category']=='TOOLS'
x_3 = GOOGLE.loc[sel_3, 'Rating']

sel_4 = GOOGLE['Category']=='FAMILY'
x_4 = GOOGLE.loc[sel_4, 'Rating']

sel_5 = GOOGLE['Category']=='FINANCE'
x_5 = GOOGLE.loc[sel_5, 'Rating']

sel_6 = GOOGLE['Category']=='ENTERTAINMENT'
x_6 = GOOGLE.loc[sel_6, 'Rating']

sel_7 = GOOGLE['Category']=='HEALTH_AND_FITNESS'
x_7 = GOOGLE.loc[sel_7, 'Rating']

sel_8 = GOOGLE['Category']=='SHOPPING'
x_8 = GOOGLE.loc[sel_8, 'Rating']

sel_9 = GOOGLE['Category']=='EDUCATION'
x_9 = GOOGLE.loc[sel_9, 'Rating']

sel_10 = GOOGLE['Category']=='GAME'
x_10 = GOOGLE.loc[sel_10, 'Rating']

Price_cats = kruskal(x_1, x_2, x_3, x_4, x_5, x_6,x_7,x_8,x_9,x_10)
# Используем криетрий Крускала-Уоллиса
# Сохраняем текстовый отчет
with open('/Users/gl.novikov/Work/Output/GOOGLE_STAT.txt', 'w') as fln:
    print('Критерий Крускала-Уоллиса для переменных \'Rating\' и \'Category\'',
          file=fln)
    print(Price_cats, file=fln)


# Графический анализ
# Анализ связи между количественной и качественной переменной
# В столбце не более трех графиков
dfn = GOOGLE.copy()

# Получение уникальных категорий из столбца "Category"  DataFrame
categories = dfn['Category'].unique().tolist()

# Разделение категорий на группы по 3 и 4
groups = [categories[:5], categories[5:]]

# Создание графика для каждой группы категорий
for index, group in enumerate(groups, start=1):
    fig, ax = plt.subplots(figsize=(12, 6))
    for category in group:
        df_subset = dfn[dfn['Category'] == category]
        df_subset.boxplot(column='Rating', ax=ax, positions=[group.index(category)])
    ax.set_xticks(range(len(group)))
    ax.set_xticklabels(group)
    ax.set_title('Категория приложения - Рейтинг')
    ax.set_ylabel('Rating')
    
    # Добавление подписей с количеством отзывов
    for i, category in enumerate(group):
        num_reviews = len(dfn[dfn['Category'] == category])
        ax.annotate(f'{num_reviews} отзывов', xy=(i, 0), xytext=(i, -0.5), ha='center', fontsize=10,
                    arrowprops=dict(arrowstyle='->', lw=1, color='black'))
    
    # Сохранение графика в PDF
    plt.savefig(f'./Graphics/Category-Rating{index}.pdf', bbox_inches='tight')
    plt.show()
    
    
#Анализ связи двух качественных переменных
import statsmodels.api as sm
# Читаем и преобразуем данные

GO = GOOGLE.copy()
# Строим таблицу сопряженности. С маргинальными частотами!!! 
crtx = pd.crosstab(GO['Category'], GO['Type'], margins=True)
# Даем имена переменным
crtx.columns.name = 'Type'
crtx.index.name = 'Category\Type'
# Из уже готовой таблицы сопряженности
# Создаем объект sm.stats.Table для проведения анализа
# В объекте находятся все необходимые статистики и дополнительные методы
tabx = sm.stats.Table(crtx)
# Альтернативный вариант создания sm.stats.Table
#table = sm.stats.Table.from_data(CA[['music', 'signal']])
# Сохраняем полученные результаты
with pd.ExcelWriter('/Users/gl.novikov/Work/Output/GOOGLE_STAT.xlsx', engine="openpyxl", 
                    if_sheet_exists='overlay', mode='a') as wrt:
# Таблица сопряженности
    tabx.table_orig.to_excel(wrt, sheet_name='Category-Type') 
    dr = tabx.table_orig.shape[0] + 2 # Смещение по строкам
# Ожидаемые частоты при независимости
    tabx.fittedvalues.to_excel(wrt, sheet_name='Category-Type', startrow=dr)
# Критерий хи квадрат для номинальных переменных
resx = tabx.test_nominal_association()
# Сохраняем результат в файле 
with open('/Users/gl.novikov/Work/Output/GOOGLE_STAT.txt', 'a') as fln:
    print('Критерий HI^2 для переменных \'Category\' и \'Type\'',
          file=fln)
    print(resx, file=fln)
# Рассчет Cramer V по формуле
nr = tabx.table_orig.shape[0]
nc = tabx.table_orig.shape[1]
N = tabx.table_orig.iloc[nr-1, nc-1]
hisq = resx.statistic
CrV = np.sqrt(hisq/(N*min((nr - 1, nc - 1))))
with open('/Users/gl.novikov/Work/Output/GOOGLE_STAT.txt', 'a') as fln:
    print('Статистика Cramer V для переменных \'Category\' и \'Type\'',
          file=fln)
    print(CrV, file=fln)



# Отбираем качественные признаки
dfn = GOOGLE.select_dtypes(include=["category"]) 

# Создаем отдельный график для каждого качественного признака
for s in dfn.columns:
    # Подсчет количества представителей каждой категории
    ftb = pd.crosstab(dfn[s], s) 
    ftb.index.name = ''
    ftb.columns.name = s
    
    # Увеличение размеров графика
    plt.figure(figsize=(12, 8))
    ftb.plot.bar(grid=True, title=s, legend=False, width=0.5)  # Увеличиваем ширину столбцов
    plt.ylabel('Количество')
    plt.xticks(rotation=45)  # Поворот подписей по оси x для лучшей читаемости
    plt.tight_layout()
    
    # Добавление значений по оси Y над каждым столбцом
    for i, v in enumerate(ftb.sum(axis=1)):
        plt.text(i, v + 3, str(v), ha='center')
    
    plt.subplots_adjust(hspace=0.7)  # Увеличение расстояния между графиками
    
    plt.savefig(f'./Graphics/google_{s}.pdf', format='pdf')  # Сохранение графика
    plt.show()  # Показ графика
"""
Графический анализ
Анализ связи между количественной целевой переменной и 
количественными объясняющими переменными
Рекомендуется зависимую переменную размещать первой или последней в 
pandas.DataFrame. Это упрощает автоматизацию расчетов.
"""
# Отбираем количественные переменные.
# Зависимая идет последней
dfn = GOOGLE.select_dtypes(include=['float64','int64'])
nrow = dfn.shape[1] - 1 # Учитываем, что одна переменная целевая - ось 'Y'
fig, ax_lst = plt.subplots(nrow, 1)
fig.figsize=(15, 9) # Дюймы
nplt = -1
for s in dfn.columns[1:]: # Последняя переменная - целевая ('Y')
    nplt += 1
    dfn.plot.scatter(s, 'Rating', ax=ax_lst[nplt], alpha=0.2)
    ax_lst[nplt].grid(visible=True)
    ax_lst[nplt].set_title(f'Связь рейтинга с {s}')
fig.subplots_adjust(wspace=0.5, hspace=1.0)




import matplotlib.pyplot as plt

dfn = GOOGLE.select_dtypes(include=['float64','int64'])
nrow = dfn.shape[1] - 1  # Учитываем, что одна переменная целевая - ось 'Y'

# Создание каждого графика рассеяния отдельно
for i, s in enumerate(dfn.columns[1:], start=1):  # Последняя переменная - целевая ('Y')
    fig, ax = plt.subplots(figsize=(15, 9))  # Устанавливаем размеры каждого графика
    dfn.plot.scatter(s, 'Rating', ax=ax, alpha=0.4)
    ax.grid(visible=True)
    ax.set_title(f'Связь рейтинга с {s}')
    plt.savefig(f'./Graphics/Rating_arg{i}.pdf', bbox_inches='tight')
    plt.show()


dfn = GOOGLE_copy.select_dtypes(include=['float64','int64'])
nrow = dfn.shape[1] - 1  # Учитываем, что одна переменная целевая - ось 'Y'

# Создание каждого графика рассеяния отдельно
for i, s in enumerate(dfn.columns[:-1], start=1):  # Последняя переменная - целевая ('Y')
    fig, ax = plt.subplots(figsize=(15, 9))  # Устанавливаем размеры каждого графика
    dfn.plot.scatter(s, 'Price', ax=ax, alpha=0.4)
    ax.grid(visible=True)
    ax.set_title(f'Связь цены с {s}')
    plt.savefig(f'./Graphics/Price_arg{i}.pdf', bbox_inches='tight')
    plt.show()
    
    

#--------------------------------------------------------------------------------------



# **********************************************************
# ============= МОДЕЛИРОВАНИЕ ==============================
# **********************************************************
import statsmodels.api as sm
# Читаем и преобразуем данные

GO = GOOGLE.copy()
GO['Category'] = GO['Category'].str.upper()
GO['Installs'] = GO['Installs'].str.upper()
GO['Type'] = GO['Type'].str.upper()
GO['Content Rating'] = GO['Content Rating'].str.upper()
GO['Last Updated'] = GO['Last Updated'].str.upper()
GO['Current Ver'] = GO['Current Ver'].str.upper()
GO['Android Ver'] = GO['Android Ver'].str.upper()
# Разбиение данных на тренировочное и тестовое множество
# frac- доля данных в тренировочном множестве
# random_state - для повторного отбора тех же элементов
GO_train = GO.sample(frac=0.8, random_state=42) 
# Символ ~ обозначает отрицание (not)
GO_test = GO.loc[~GO.index.isin(GO_train.index)] 

# Будем накапливать данные о качестве постреонных моделей
# Используем  adjR^2 и AIC
mq = pd.DataFrame([], columns=['adjR^2', 'AIC']) # Данные о качестве

"""
Построение базовой модели
Базовая модель - линейная регрессия, которая включает в себя 
все количественные переменные и фиктивные переменные дял качественных 
переменных с учетом коллинеарности. Для каждого качетсвенного показателя
включаются все уровни за исключением одного - базового. 
"""
# Формируем целевую переменную
Y = GO_train['Price']
# Формируем фиктивные (dummy) переменные для всех качественных переменных

DUM = pd.get_dummies(GO_train[['Category','Installs', 'Type', 'Content Rating', 'Last Updated', 'Current Ver', 'Android Ver']])

# Выбираем переменные для уровней, которые войдут в модель
# Будет исключен один - базовый. Влияние включенных уровней на зависимую 
# переменную отсчитывается от него
del DUM['Category_BOOKS_AND_REFERENCE']
del DUM['Installs_50,000,000+']
del DUM['Type_FREE']
del DUM['Content Rating_ADULTS ONLY 18+']
del DUM['Last Updated_2018']
del DUM['Current Ver_V.9']
del DUM['Android Ver_V.8 AND UP']

# Формируем pandas.DataFramee содержащий матрицу X объясняющих переменных 
# Добавляем слева фиктивные переменные
X = pd.concat([DUM, GO_train[['Rating', 'Reviews','Size']]], axis=1)
# Добавляем переменную равную единице для учета константы
X = sm.add_constant(X)
X = X.astype({'const':'uint8'}) # Сокращаем место для хранения константы
# Формируем объект, содержащй все исходные данные и методы для оценивания
linreg00 = sm.OLS(Y, X.astype(float))
# Оцениваем модель
fitmod00 = linreg00.fit()
# Сохраняем результаты оценки в файл
with open('./Output/GOOGLE_STAT.txt', 'a') as fln:
    print('\n ****** Оценка базовой модели ******',
          file=fln)
    print(fitmod00.summary(), file=fln)


# Проверяем степень мультиколлинеарности только базовой модели
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame() # Для хранения 
X_q = X.select_dtypes(include=['float64','int64'])# Только количественные регрессоры
vif["vars"] = X_q.columns
vif["VIF"] = [variance_inflation_factor(X_q.values, i) 
              for i in range(X_q.shape[1])]
# Сохраняем полученные результаты
with pd.ExcelWriter('./Output/GOOGLE_STAT.xlsx', engine="openpyxl", 
                    if_sheet_exists='overlay', mode='a') as wrt:
    vif.to_excel(wrt, sheet_name='vif')

X1 = X.copy()
del X1['Rating']
del X1['Reviews']
# Проверяем гетероскедастичность базовой модели
# помощью коритерия White(а) и F критерия
from statsmodels.stats.diagnostic import het_white
e = fitmod00.resid
WHT = pd.DataFrame(het_white(e, X1), index= ['LM', 'LM_P', 'F', 'F_P'])
# Сохраняем полученные результаты
with pd.ExcelWriter('./Output/GOOGLE_STAT.xlsx', engine="openpyxl", 
                    if_sheet_exists='overlay', mode='a') as wrt:
    WHT.to_excel(wrt, sheet_name='het')

# Сохраняем данные о качестве модели
q = pd.DataFrame([
    [fitmod00.rsquared, fitmod00.rsquared_adj, fitmod00.aic]
    ], 
    columns=['R^2', 'adjR^2', 'AIC'], 
    index=['base_00']
)

mq = pd.concat([mq, q])    



# ****************** Проверка гипотез ******************

"""
Сила влияния оценки приложения на цену зависит от количества оценок. 
Чем больше оценок тем влияние больше.

"""
# Вводим переменную взаимодействия
X_1 = X.copy()
X_1['Reviews_with_Rating'] = X_1['Reviews']*X_1['Rating']
linreg02 = sm.OLS(Y,X_1.astype(float))
fitmod02 = linreg02.fit()
# Сохраняем результаты оценки в файл
with open('./Output/GOOGLE_STAT.txt', 'a') as fln:
    print('\n ****** Оценка базовой модели ******',
          file=fln)
    print(fitmod02.summary(), file=fln)
    
# Сохраняем данные о качестве модели
q = pd.DataFrame([fitmod02.rsquared,fitmod02.rsquared_adj, fitmod02.aic], 
                 index=['R^2','adjR^2', 'AIC'], columns=['hyp_01']).T
mq = pd.concat([mq, q])    

# Гипотеза не отвергается. Модель стала заметно лучше.

"""
Сила влияния размера на цену приложения зависит от типа приложения. 
Если тип приложения функциональный (TOOLS), то влияние больше.

"""
# Вводим переменную взаимодействия
X_2 = X.copy()
X_2['Size_with_TOOLS'] = X_2['Size']*X_2['Category_TOOLS']

linreg03 = sm.OLS(Y,X_2.astype(float))
fitmod03 = linreg03.fit()
# Сохраняем результаты оценки в файл
with open('./Output/GOOGLE_STAT.txt', 'a') as fln:
    print('\n ****** Оценка базовой модели ******',
          file=fln)
    print(fitmod03.summary(), file=fln)
    
# Сохраняем данные о качестве модели
q = pd.DataFrame([fitmod03.rsquared, fitmod03.rsquared_adj, fitmod03.aic], 
                 index=['R^2', 'adjR^2', 'AIC'], columns=['hyp_02']).T
mq = pd.concat([mq, q])   

#**************** Оптимизация итоговой модели ****************
X_itog = X.copy()
# Используем регулярное выражение для поиска столбцов, начинающихся на 'Android'
cols_to_drop = X_itog.columns[X_itog.columns.str.startswith('Android') 
                              | (X_itog.columns.str.startswith('Last Updated')) 
                              | (X_itog.columns.str.startswith('Current')) 
                              | (X_itog.columns.str.startswith('Content Rating')) 
                              | (X_itog.columns.str.startswith('Installs'))]
# Удаляем эти столбцы
X_itog.drop(cols_to_drop, axis=1, inplace=True)

X_itog['Reviews_with_Rating'] = X_itog['Reviews']*X_itog['Rating']
linreg04 = sm.OLS(Y,X_itog.astype(float))
fitmod04 = linreg04.fit()
# Сохраняем результаты оценки в файл
with open('./Output/GOOGLE_STAT.txt', 'a') as fln:
    print('\n ****** Оценка базовой модели ******',
          file=fln)
    print(fitmod04.summary(), file=fln)
# Сохраняем данные о качестве модели
q = pd.DataFrame([fitmod04.rsquared, fitmod04.rsquared_adj, fitmod04.aic], 
                 index=['R^2','adjR^2', 'AIC'], columns=['itog']).T
mq = pd.concat([mq, q])   


#*********** Предсказательная сила *********

# Копирование данных
GO = GOOGLE.copy()
columns_to_normalize = ['Category', 'Installs', 'Type', 'Content Rating', 'Last Updated', 'Current Ver', 'Android Ver']
for column in columns_to_normalize:
    GO[column] = GO[column].str.upper()

# Разбиение данных на тренировочное и тестовое множество
GO_train = GO.sample(frac=0.8, random_state=42) 
GO_test = GO.loc[~GO.index.isin(GO_train.index)]

# Формирование полного списка уникальных категорий
all_data = pd.concat([GO_train, GO_test])
DUM_full = pd.get_dummies(all_data[['Category', 'Installs', 'Type', 'Content Rating', 'Last Updated', 'Current Ver', 'Android Ver']], drop_first=True)

# Применение списка к тренировочным и тестовым данным
DUM_train = DUM_full.loc[GO_train.index]
DUM_test = DUM_full.loc[GO_test.index]

# Объединение дамми-переменных с количественными переменными
X_train = pd.concat([DUM_train, GO_train[['Rating', 'Reviews', 'Size']]], axis=1)
X_train = sm.add_constant(X_train)
Y_train = GO_train['Price']

X_test = pd.concat([DUM_test, GO_test[['Rating', 'Reviews', 'Size']]], axis=1)
X_test = sm.add_constant(X_test)
Y_test = GO_test['Price']

# Обучение модели линейной регрессии
linreg = sm.OLS(Y_train, X_train.astype(float)).fit()

# Предсказание на тестовых данных
pred_ols = linreg.get_prediction(X_test.astype(float))
frm = pred_ols.summary_frame(alpha=0.05)

# Генерация доверительных интервалов
iv_l = frm["obs_ci_lower"]
iv_u = frm["obs_ci_upper"]
fv = frm['mean']

# Построение графиков
name = 'Rating'
Z = X_test.loc[:, name]
dfn = pd.DataFrame([Z, Y_test, fv, iv_u, iv_l]).T
dfn = dfn.sort_values(by=name)
fig, ax = plt.subplots(figsize=(8, 6))
for z in dfn.columns[1:]:
    dfn.plot(x=dfn.columns[0], y=z, ax=ax)
ax.legend(loc="best")
plt.show()

# Расчет ошибок
dif = np.sqrt((dfn.iloc[:,1] - dfn.iloc[:,2]).pow(2).sum()/dfn.shape[0])
mn = dfn.iloc[:,1].sort_index()
out = ((mn > iv_u) + (mn < iv_l)).sum()/dfn.shape[0]

errors0 = pd.DataFrame([np.sqrt(np.mean((linreg.predict(X_test) - Y_test)**2)), 
                      np.mean(abs(linreg.predict(X_test) - Y_test))],
                     index=['Среднеквадратическая погрешность',
                            'Абсолютная погрешность'], columns=['Начальная модель']).T


# Создание переменной взаимодействия
GO_train['Reviews_with_Rating'] = GO_train['Reviews'] * GO_train['Rating']
GO_test['Reviews_with_Rating'] = GO_test['Reviews'] * GO_test['Rating']

# Объединение дамми-переменных с количественными переменными
X_train = pd.concat([DUM_train, GO_train[['Rating', 'Reviews', 'Size', 'Reviews_with_Rating']]], axis=1)
X_train = sm.add_constant(X_train)
Y_train = GO_train['Price']

X_test = pd.concat([DUM_test, GO_test[['Rating', 'Reviews', 'Size', 'Reviews_with_Rating']]], axis=1)
X_test = sm.add_constant(X_test)
Y_test = GO_test['Price']

# Обучение модели линейной регрессии
linreg02 = sm.OLS(Y_train, X_train.astype(float)).fit()

# Предсказание на тестовых данных
pred_ols02 = linreg02.get_prediction(X_test.astype(float))
frm02 = pred_ols02.summary_frame(alpha=0.05)

# Генерация доверительных интервалов
iv_l02 = frm02["obs_ci_lower"]
iv_u02 = frm02["obs_ci_upper"]
fv02 = frm02['mean']

# Построение графиков
fig, ax = plt.subplots(figsize=(8, 6))
dfn02 = pd.DataFrame({'Rating': X_test['Rating'], 'Observed Price': Y_test, 'Predicted Price': fv02, 'Upper CI': iv_u02, 'Lower CI': iv_l02})
dfn02 = dfn02.sort_values(by='Rating')
dfn02.plot(x='Rating', y=['Observed Price', 'Predicted Price', 'Upper CI', 'Lower CI'], ax=ax)
ax.legend(['Наблюдаемая цена', 'Предсказанная цена', 'Верхний ДИ', 'Нижний ДИ'])
plt.show()

# Расчет ошибок
errors02 = pd.DataFrame({
    'Среднеквадратическая ошибка': [np.sqrt(np.mean((linreg02.predict(X_test) - Y_test)**2))],
    'Средняя абсолютная ошибка': [np.mean(abs(linreg02.predict(X_test) - Y_test))]}, index=['Промежуточая модель'])


# Объединение дамми-переменных с количественными переменными
X_train = pd.concat([DUM_train, GO_train[['Rating', 'Reviews', 'Size']]], axis=1)
Y_train = GO_train['Price']
X_train = sm.add_constant(X_train)

X_test = pd.concat([DUM_test, GO_test[['Rating', 'Reviews', 'Size']]], axis=1)
Y_test = GO_test['Price']
X_test = sm.add_constant(X_test)

# Создание новой модели с удалением столбцов и добавлением переменной взаимодействия
cols_to_drop = X_train.columns[X_train.columns.str.startswith('Android') 
                               | X_train.columns.str.startswith('Last Updated')
                               | X_train.columns.str.startswith('Current')
                               | X_train.columns.str.startswith('Content Rating')
                               | X_train.columns.str.startswith('Installs')]
X_train_final = X_train.drop(cols_to_drop, axis=1)
X_train_final['Reviews_with_Rating'] = X_train_final['Reviews'] * X_train_final['Rating']

X_test_final = X_test.drop(cols_to_drop, axis=1)
X_test_final['Reviews_with_Rating'] = X_test_final['Reviews'] * X_test_final['Rating']

# Обучение модели
linreg_final = sm.OLS(Y_train, X_train_final.astype(float)).fit()

# Предсказание на тестовых данных
pred_ols_final = linreg_final.get_prediction(X_test_final.astype(float))
frm_final = pred_ols_final.summary_frame(alpha=0.05)

# Генерация доверительных интервалов
iv_l_final = frm_final["obs_ci_lower"]
iv_u_final = frm_final["obs_ci_upper"]
fv_final = frm_final['mean']

# Построение графиков
fig, ax = plt.subplots(figsize=(8, 6))
dfn_final = pd.DataFrame({'Rating': X_test_final['Rating'], 'Observed Price': Y_test, 'Predicted Price': fv_final, 'Upper CI': iv_u_final, 'Lower CI': iv_l_final})
dfn_final = dfn_final.sort_values(by='Rating')
dfn_final.plot(x='Rating', y=['Observed Price', 'Predicted Price', 'Upper CI', 'Lower CI'], ax=ax)
ax.legend(['Наблюдаемая цена', 'Предсказанная цена', 'Верхний ДИ', 'Нижний ДИ'])
plt.show()

# Расчет ошибок
errors_final = pd.DataFrame({
    'Среднеквадратическая ошибка': [np.sqrt(np.mean((linreg_final.predict(X_test_final) - Y_test)**2))],
    'Средняя абсолютная ошибка': [np.mean(abs(linreg_final.predict(X_test_final) - Y_test))],}, index=['Финальная модель'])

"""
dir(CB)
help(CB.xs)
CB.dtypes
"""




