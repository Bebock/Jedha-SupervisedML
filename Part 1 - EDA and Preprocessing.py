#!/usr/bin/env python
# coding: utf-8

# In[39]:


import warnings
warnings.filterwarnings('ignore')

import math
import pandas as pd
import numpy as np
from scipy import stats
from math import ceil
from pickle import FALSE
import matplotlib.ticker as ticker
from bioinfokit.analys import stat
#from pandas_profiling import ProfileReport

# Graphics

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

# Preprocessing 

import missingno as msno
from fancyimpute import IterativeImputer as MICE
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.compose import ColumnTransformer

# Model Selection & evaluation

from mlxtend.evaluate import bias_variance_decomp
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression, BayesianRidge, ElasticNet
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score
from yellowbrick.regressor import ResidualsPlot

#pio.renderers.default = 'vscode'
pio.templates.default = 'plotly'
pio.renderers.default = "notebook"


# # Part 1 : Exploratory Descriptive Analysis (EDA) and Data Preprocessing

# In[40]:


# Data import 
data = pd.read_csv('Walmart_Store_sales.csv')

# Overview 
len0 = len(data)
data.describe(include='all')


# # 1. First preprocessing
# 
# ## 1.1 Variable Date

# ### First glance on dates

# In[41]:


# Converting date in datetime type
data.Date = pd.to_datetime(data['Date'])

# Creating a new dataframe summing WS per day
df = data[['Weekly_Sales', 'Date']].groupby('Date').sum('Weekly_Sales')

# Reindexing the new dataframe with the full range of dates
Dates = pd.date_range(df.index.min(), df.index.max())
df = df.reindex(Dates, fill_value = np.nan)


# In[42]:


# Time series Plot 
plt.figure(figsize = (13,7))
fig = sns.barplot(Dates, df.Weekly_Sales)
new_ticks = [i.get_text()[0:10] for i in fig.get_xticklabels()]
plt.xticks(range(0, len(new_ticks), 20), new_ticks[::20], rotation=45)
plt.tight_layout()


# We can see that we do not have data for each date covered by the period. We can not analyze the data with Time Series and we have to keep in mind the parcellar nature of the data that we have to build the model, as it will definitively impair its quality.
# 
# Instead of treating date as it is, we will extract some information from this variable, to add potentially interesting variables to the model. 

# ### Creating usable features from the *Date* column

# 
# The *Date* column cannot be included as it is in the model. We will create new columns that contain the following numeric features : 
# - *year*
# - *month*
# - *day*
# - *day of week*

# In[43]:


print('Number of missing data in Date : ' + str(data['Date'].isnull().sum()))
print('Proportion of missing data in Date : ' + str(data['Date'].isnull().sum() / len(data) * 100))


# In[44]:


# Extraction of date features
data['year'] = data.Date.dt.year
data['month'] = data.Date.dt.month
data['day'] = data.Date.dt.day
data['weekday'] = data.Date.dt.weekday

def week_of_month(dt):
    """ 
    Returns the week of the month for the specified date.
    """
    if (dt is pd.NaT) == False : 
        first_day = dt.replace(day=1)
        dom = dt.day
        adjusted_dom = dom + first_day.weekday()
        return int(ceil(adjusted_dom/7.0))
    else :
        return np.nan
data['week'] = data.Date.apply(week_of_month)

# Deleting the column Date
data = data.drop(['Date'], axis = 1)


# In[45]:


# Barplot of each qualitative variable
cat_features = ['year', 'month', 'day', 'weekday', 'week']

fig2 = make_subplots(rows = math.ceil(len(cat_features)/3), cols = 3, subplot_titles = cat_features)
for i in range(len(cat_features)):
    
    x_coords = data[cat_features[i]].value_counts().index.tolist()
    y_coords = data[cat_features[i]].value_counts().tolist()

    fig2.add_trace(
        go.Bar(
            x = x_coords,
            y = y_coords),
        row = i//3 + 1,
        col = i - (i//3 * 3) + 1)
fig2.update_layout(
        title = go.layout.Title(text = "Barplot of qualitative variables", x = 0.5), showlegend = False, 
            autosize=True)
fig2.show()


# We can observe that : 
# 
# * We have a decreasing amount of data with year
# * The amount of data is not equivalent according to month with lower levels in January and during Automn. 
# * Curiously, none of the entries were done on a 21th of any month
# * Weekdays are mainly equal to Friday (more than 80% of the dates are Friday) : As the response variable is WEEKLY sales, it can make sense that we have one entry per week in each store. In that case, weekdays, and by extension days of the month are not relevant here because of this specific time frame based on weeks. After checking in the original data source (Kaggle competition - https://www.kaggle.com/datasets/yasserh/walmart-dataset), Date is defined as "The week of sales", so the Date we have here is not refering to a day but to a week.
# * Finally, regarding amount of entries according to the week of the month, we have a little more data in the 3rd week of the month
# 
# This crude analysis confirms the parcellar nature of the dataset we have to analyze. All entries for the covered period are not present in our dataset as the original Kaggle dataset contained 6435 rows against 150 here. 

# In[46]:


data = data.drop(['weekday', 'day'], axis = 1)


# ## 1.2. Removing lines
# 
# ### Drop lines where target values are missing
# 
#  - Here, the target variable (Y) corresponds to the column *Weekly_Sales*
#  - We never impute the target : it might create some bias in the predictions.
#  - Then, we will just drop the lines in the dataset for which the value in *Weekly_Sales* is missing.

# In[47]:


data = data.dropna( how='any',
                    subset=['Weekly_Sales'])
lenY = len(data)


# ### Drop lines with too many NA

# In[48]:


# lines with too many NA
plt.hist((data.isnull().sum(axis=1) / 9 * 100).sort_values(ascending = False))
plt.title('Distribution of the percentages of missing values in a line')
plt.xlabel('Percent of missing values in a line')
plt.ylabel('Frequency');


# In[49]:


# Deleting rows with 40% or more of missing values
data = data[data.isnull().sum(axis=1) / 9 < 0.4]
lenRow = len(data)


# ### Drop duplicated rows

# In[50]:


# Removes duplicate rows based on all columns.
data = data.drop_duplicates()
lenDup = len(data)


# ### Drop lines with invalid values or outliers :
# 
# In this project, will be considered as outliers all the numeric features that don't fall within the range : $[\bar{X} - 3\sigma, \bar{X} + 3\sigma]$. 
# 
# This concerns the columns : *Temperature*, *Fuel_price*, *CPI* and *Unemployment*

# In[51]:


for c in ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment'] : 
    print(c, " ", data[c][(data[c] < data[c].mean() - 3 * data[c].std()) | (data[c] > data[c].mean() + 3 * data[c].std())].count())


# Unemployement contains outliers that are removed.

# In[52]:


data = data.drop(data[(data['Unemployment'] < data['Unemployment'].mean() - 3 * data['Unemployment'].std()) | 
                      (data['Unemployment'] > data['Unemployment'].mean() + 3 * data['Unemployment'].std())].index)
lenOut = len(data)


# In[53]:


# Checks 
print(max(np.abs(stats.zscore(data.Unemployment,nan_policy='omit'))))


# ### Summary

# In[54]:


print("After this forst preprocessing, we deleted :")
print('')

print(len0 - lenY, " rows with missing Weekly Sales.")
print(lenY - lenRow, " rows with too many missing values.")
print(lenRow - lenDup, " duplicated rows.")
print(lenDup - lenOut, " row with outliers.")
print('')

print('So, we deleted ', len0 - lenOut, " rows and finally have a dataframe of ", lenOut, " rows")


# # 2. Exploratory Descriptive Analysis (EDA) and first features selection

# ## 2.1. Univariate analysis - Variables Description

# ### Weekly Sales

# In[55]:


plt.figure(figsize = [20, 7])
# left histogram: data plotted in natural units
plt.subplot(1, 4, 1)
plt.hist(data.Weekly_Sales)
plt.xlabel('Weekly sales')
# right histogram: data plotted after direct log transformation
plt.subplot(1, 4, 2)
log_data = np.log10(data.Weekly_Sales) # direct data transform
plt.hist(log_data)
plt.xlabel('log(Weekly sales)')
plt.subplot(1, 4, 3)
pt = PowerTransformer(method='yeo-johnson', standardize=True)
plt.hist(pd.DataFrame(pt.fit_transform(data['Weekly_Sales'].values.reshape(-1,1))))
plt.xlabel('YJ(Weekly sales)')
plt.subplot(1, 4, 4)
pt = PowerTransformer(method='box-cox', standardize=True)
plt.hist(pd.DataFrame(pt.fit_transform(data['Weekly_Sales'].values.reshape(-1,1))))
plt.xlabel('Box-Cox(Weekly sales)');


# The variable we want to predict, Weekly Sales, is not normally distributed and we can not find a satisfying transformation because of its bi-modality. We will have to carefully inspect the regression residuals to make sure that this non normality does not compromise the model quality and the required assumptions of this kind of model (parametric). 

# ### Quantitative variables

# In[56]:


# Distribution of each numeric variable
num_features = ['Temperature', 'Fuel_Price', 'Unemployment', 'CPI']

fig1 = make_subplots(cols = len(num_features), rows = 1, subplot_titles = num_features)
for i in range(len(num_features)):
    fig1.add_trace(
        go.Histogram(
            x = data[num_features[i]], nbinsx = 10),
        col = i + 1,
        row = 1)
fig1.update_layout(
        title = go.layout.Title(text = "Distribution of quantitative variables", x = 0.5), 
        showlegend = False, 
        autosize=True)
fig1.show()


# We can observe that : 
# 
# - Fuel_price is bimodal, 
# - Temperature and Unemployment looks like relatively normal distributions
# - CPI does not appear to be "continuous" in terms of the values the variable could take. According to its distribution, it is questionable to try to establish a continuous relationship between this variable and Weekly_Sales. We decided to categorize the CPI variable, recoding it into 3 categories.  

# In[57]:


# Creating the bin width
bins = pd.qcut(data['CPI'][data['CPI'] > 160], 2, retbins=True)[1].tolist()

# Recoding CPI into CPI_R
data['CPI_R'] =  data['CPI'].apply(lambda x: 0 if x < 160 else (1 if x <= math.floor(bins[1]) else 2))


# In[58]:


# Plot of the 2 variables : the continuous version and the categorical one. 
sns.histplot(data=data, x="CPI", hue="CPI_R")

# Removing CPI 
data = data.drop(['CPI'], axis = 1)
num_features.remove('CPI')
num_features = ['Temperature', 'Fuel_Price', 'Unemployment']


# ### Qualitative variables

# In[59]:


# Recoding
recodages = {"Holiday_Flag":     {0: 'Regular', 1 : 'Holiday'}}
data = data.replace(recodages)


# In[60]:


# Barplot of each qualitative variable
cat_features = ['Store', 'Holiday_Flag', 'CPI_R', 'year', 'month', 'week']

fig2 = make_subplots(rows = math.ceil(len(cat_features)/3), cols = 3, subplot_titles = cat_features)
for i in range(len(cat_features)):
    
    x_coords = data[cat_features[i]].value_counts().index.tolist()
    y_coords = data[cat_features[i]].value_counts().tolist()

    fig2.add_trace(
        go.Bar(
            x = x_coords,
            y = y_coords),
        row = i//3 + 1,
        col = i - (i//3 * 3) + 1)
fig2.update_layout(
        title = go.layout.Title(text = "Barplot of qualitative variables", x = 0.5), showlegend = False, 
            autosize=True)
fig2.show()


# As already mentionned, the time variables are not balanced in terms of the relative frequency of their modalities, making us think that the data are a parcellar sample of a more complete dataset. 
# 
# Regarding, the 3 other categorical variables, we can see that : 
# - Holiday Flag counts a vast majority of regular date/week 
# - CPI is relativement homogeneous according to the way we built it
# - Store is truly unbalanced with some stores completely under-represented in the dataset

# In[61]:


cat_features = ['Store', 'Holiday_Flag', 'CPI_R', 'year', 'month']


# ## 2.2 Bivariate analysis
# 
# ### Global Overview

# In[62]:


# Pair plot of all variables
sns.pairplot(data, corner = True);


# In[63]:


# Heatmap - Correlation matrix
plt.figure(figsize = (6.5,6.5))
sns.heatmap(data.corr(), 
            xticklabels = data.corr().columns, 
            yticklabels = data.corr().columns, 
            vmin = -1, 
            vmax = 1, 
            annot = True);


# In[64]:


# Create correlation matrix
corr_matrix = data.corr().abs()

#Identification of high correlations with potential colinearity issues :
correlations = corr_matrix.unstack().reset_index()
correlations = correlations.rename(columns=dict(zip(correlations.columns, ['Variable 1', 'Variable 2', 'Correlation'])))
correlations[(correlations.Correlation !=1) &  (abs(correlations.Correlation) > 0.8)].drop_duplicates(subset=['Correlation'])

########### Delete automatique #############
# Select upper triangle of correlation matrix
#upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
# Find features with correlation greater than 0.95
#to_drop = [column for column in upper.columns if any(upper[column] > 0.80)]
#print(to_drop)
# Drop features 
#data.drop(to_drop, axis=1, inplace=True)


# From these first crude bivariate analyses, and keeping in mind that we are considering, to ease this first and gloabl approach, qualitative variables as quantitative ones, we can conclude that :
# 
# - Fuel Price and Year are strongly positively related, so we will have to choose one of these 2 variables to be included in the model, to avoid any mutlicolinearity (OR we will have to use regularization to handle multicolinearity by itself). 
# - Curiously, we do not observe any correlation between month of the year (seasons) and temperature. As we do not have any information avout the store location, it is quite difficult to investigate this non relationship but we can hypothesize that it is due to the geographical and thus climatic differences between the stores. 
# - Quantitative variables globally assymetrical and qualitative ones are globally unbalanced, NOT putting us in the best conditions to perform a linear regression on a small size dataset. 
# - Weekly Sales do not appear to be strongly related to any of the variables included in the dataset. The highest correlation that we can observe is with CPI (r = -0.24) which is not considered as a strong correlation. Considering this absence of strong bivariate correlations, we can expect some difficulties in terms of accurately modeling Weekly Sales with the variables in our hand. 

# ### Relationships with Weekly Sales
# 
# #### Quantitative variables

# In[65]:


# Pair plot
sns.pairplot(data[num_features + ['Weekly_Sales']], corner = True, height = 1.5);


# In[66]:


# Correlation matrix
corr_matrix = data[['Weekly_Sales'] + num_features].corr()
sns.heatmap(corr_matrix, cmap="YlGnBu", annot=True);


# As we said, at the first sight, none of the quantitative variables seems to be related to Weekly_Sales. 
# 
# First of all, Weekly sales and Fuel Price are absolutely not correlated (r = -0.027) and the examination of the dot plot could not lead us to think that maybe a more complicated and non linear relationship could exist and not be catched by the linear correlation coefficient. As we saw that we have a strong correlation between fuel price and Year, we chose to not consider fuel price anymore, because this variable will not be helpful to exmplain and predict Weekly Sales and it could impair the model due to its relationship with another variable of the model. 
# 
# Before removing this variable from the dataset, just a check to be sure that what we observe is not related to the bimodality af the variable. 

# In[67]:


data['FP_R'] = pd.qcut(data['Fuel_Price'], 2)
data.groupby('FP_R')['Weekly_Sales'].mean().plot(kind='bar');


# The mean Weekly Sales is the same when the fuel price is low (below median price) or high (above median price). Let's remove this variable

# In[68]:


data = data.drop(['Fuel_Price', 'FP_R'], axis = 1)
num_features.remove('Fuel_Price')


# Let's now have a closer look at the link between Weekly Sales and Temperature (r = - 0.18) and Unemployment (r = 0.17)

# In[69]:


fig, axes = plt.subplots(1, 2)
sns.regplot(x='Temperature', y='Weekly_Sales', data=data, ax=axes[0])
sns.regplot(x='Unemployment', y='Weekly_Sales', data=data, ax=axes[1])
fig.tight_layout()
plt.show();


# The small correlations indicate that the lower temperature is and the higher the unemployment rate is and the higher are the weekly sales. However, let's note that the correlation coefficients are really small and that the plots indicate a poor adjustement of the data by these 2 relationships. 

# #### Qualitative variables

# In[70]:


cat_features = ['Holiday_Flag', 'CPI_R', 'year', 'week', 'month', 'Store']

# creating grid for subplots
fig = plt.figure()
fig.set_figheight(10)
fig.set_figwidth(15)
 
ax1 = plt.subplot2grid(shape=(2, 4), loc=(0, 0), colspan=1)
ax2 = plt.subplot2grid(shape=(2, 4), loc=(0, 1), colspan=1)
ax3 = plt.subplot2grid(shape=(2, 4), loc=(0, 2), colspan=1)
ax4 = plt.subplot2grid(shape=(2, 4), loc=(0, 3), colspan=1)

ax5 = plt.subplot2grid(shape=(2, 4), loc=(1, 0), colspan=2)
ax6 = plt.subplot2grid(shape=(2, 4), loc=(1, 2), colspan=2)

axes = [ax1, ax2, ax3, ax4, ax5, ax6] 

# plotting subplots
for i in range(len(cat_features)) :
    axe = 'ax' + str(i)
    sns.boxplot(x=cat_features[i], y='Weekly_Sales', data=data, ax = axes[i])
     
# automatically adjust padding horizontally
# as well as vertically.
plt.tight_layout()
 
# display plot
plt.show()


# In[71]:


for var in cat_features : 
    res = stat()
    formula = 'Weekly_Sales ~ C(' + var + ')'
    res.anova_stat(df = data, res_var = 'Weekly_Sales', anova_model = formula)
    print('Effect of {} - p value  = {}'.format(var, round(res.anova_summary['PR(>F)'][0],3)))


# Regarding the qualitative variables, we can observe significant variations of Weekly sales according to the store and to the CPI. For the Store effect, it could have been really interesting for the model to have more data for each store or to be able to group them, for example by geographical area.  
# 
# It is quite surprising to not observe any significant time effect and especially Month effect as we could expect few typical months to be associated with a sales volume augmentation (eg. November with Thanksgiving and December with Christmas and the winter holidays). 
# 
# To better understand this lack of effect, let's try to see if the effect of Month is hidden in years. 

# In[72]:


def year_graph(year):
  df = data[data.year == year].groupby(['month'])['Weekly_Sales'].mean().to_frame()
  df.Weekly_Sales = [round(x / 1000) for x in df.Weekly_Sales]
  
  fig, ax = plt.subplots(figsize=(10,2))
  ax.bar(df.index, df["Weekly_Sales"])
  ax.set_title("Evolution of weekly sales means per month (K$) - Year {}".format(int(year)))
  ax.set_xlabel("Month")
  ax.set_ylabel("Mean of weekly sales (K$)");

  for c in ax.containers:
    ax.bar_label(c, label_type='center', color="white")
  plt.show()

for year in list(set(data.year[data.year.isnull() == False])):
  year_graph(year) 


# In[73]:


weekly_sales_2010 = data[data.year==2010]['Weekly_Sales'].groupby(data['month']).mean()
weekly_sales_2011 = data[data.year==2011]['Weekly_Sales'].groupby(data['month']).mean()
weekly_sales_2012 = data[data.year==2012]['Weekly_Sales'].groupby(data['month']).mean()
plt.figure(figsize=(10,5))
sns.lineplot(weekly_sales_2010.index, weekly_sales_2010.values)
sns.lineplot(weekly_sales_2011.index, weekly_sales_2011.values)
sns.lineplot(weekly_sales_2012.index, weekly_sales_2012.values)
plt.grid()
plt.xticks(np.arange(1, 12, step = 1))
plt.legend(['2010', '2011', '2012'], loc = 'best', fontsize = 10)
plt.title('Average Weekly Sales registered by month - Per Year', fontsize = 12)
plt.ylabel('Sales', fontsize = 10)
plt.xlabel('Week', fontsize = 10)
plt.show()


# The year 2012 seems to have lower sales volume compared to the other years however, the fact that this year had a full month missing (August) leads to think that again we face a very parcellar dataset, and that 2012 could be less documented / exhaustive than 2010 and 2011. The fact that the seasonal trend looks completely different from a year to another continues to reinforce the idea of a dataset very incomplete. 
# 
# Let's examine if the temporal trend could be more meaningful if we decompose the data by store. 

# In[88]:


Stores = data.groupby([data.Store, data.year])['Weekly_Sales'].agg('mean').unstack()
Stores = Stores.apply(lambda x: round(x / 1000, 0))
Stores.index = [int(x) for x in Stores.index.tolist()]

ax = Stores.plot(kind = 'bar', figsize = (15,7));
for c in ax.containers:
    ax.bar_label(c, rotation = 90, fontsize = 8)
plt.legend(['2010', '2011', '2012'], loc = 'best', fontsize = 9)
plt.title('Ventes annuelles en K$ par magasin et par année', fontsize = 12)
plt.show()


# The yearly trend is not the same for each store, with the best and the worst years differing from one store to another. Furthermore, all the stores do not have data for the 3 years and we cann not think to any store closure as for some (10, 17), 2011 is missing with 2010 and 2012 documented. Again, these insights reinforce the idea of a dataset incomplete, with a quite porrt quality, that we have to keep in mind for our conclusions. 

# In[ ]:


data.to_csv('preprocessed.csv', index = False)

