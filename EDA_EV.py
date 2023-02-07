
## Performing EDA on the data collected in data_prep.py

#Loading file
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style = "white")

#Importing initial dataframe

EV_sales = pd.read_csv('C:/Users/chaub/Documents/CU_Boulder/Spring 2023/CSCI 5622 Machine Learning/CSCI5662 Project/Initial_EV_dataframe.csv')
EV_sales.head()


sns.histplot(data = EV_sales['Label'])



###Plot 1 to check how many vehicles are registered in each state

gfg = sns.boxplot(x ="day", y ="total_bill", data = tips)
 
# add label to the axis and label to the plot
gfg.set(xlabel ="GFG X", ylabel = "GFG Y", title ='some title')

sns.set(style = 'whitegrid')
plt.figure(figsize=(6, 6), 
           dpi = 600)
sns.set(font_scale=0.5)
plot_1 = sns.barplot(x = EV_sales['State'], y= EV_sales['Label_ref'])
plot_1.set(xlabel = 'State', ylabel= 'Number of EV registered', title = 'Electric Vehicle registrations by state 2008-2020')
plt.xticks(rotation = 90)

###############################################################################

data = pd.read_csv('C:/Users/chaub/Documents/CU_Boulder/Spring 2023/CSCI 5622 Machine Learning/CSCI5662 Project/ElectricVehiclePred.csv')
data.head()

data.columns
sns.violinplot(data=data['Percentage_pop_with_bachelors'])

sns.lineplot(x = data['Year'], y = data['Adults 26-34'])
sns.lineplot(x = data['Year'], y = data['Adults 19-25'])
sns.lineplot(x = data['Year'], y = data['Adults 35-54'])

sns.scatterplot(x=data['Adults 26-34'], y=data['num_jobs_tot'], hue = data['Label'])

### Plot 2 to check how many vehicles are registered in each state
sns.set(style = 'white')
plt.figure(figsize=(6, 6), 
           dpi = 600)
sns.set(font_scale=0.5)
plot_2 = sns.barplot(x = data['State'], y = data['Number of registrations'])
plot_2.set(xlabel = 'State', ylabel= 'Number of EV registered', title = 'Electric Vehicle registrations by state 2008-2020')
plt.xticks(rotation = 90)


### Plot 3 check if bachelor attainment level compares with Electric vehicles. 

sns.set(style = 'whitegrid')
plt.figure(figsize =(6,6),
           dpi = 600)
sns.set(font_scale=0.5)
plot_3 = sns.histplot(x = 'Percentage_pop_with_bachelors', y = 'State',  data = data, kde=True)
plot_3.set(xlabel = 'Percentage of population with a Bachelors degree', ylabel = 'State', title = 'Percentage of population with a bachelors in USA 2008-2020' )
plot_3

### Plot  Plotting EV registrations for each state
min_reg = data['Number of registrations'].min()
max_reg = data['Number of registrations'].max()

range_num = max_reg-min_reg

data['norm_registrations'] = (data['Number of registrations']-data['Number of registrations'].min())*100/(data['Number of registrations'].max()-data['Number of registrations'].min())
data.head()

sns.set(style = 'white')
plt.figure(figsize=(6, 6), 
           dpi = 600)
sns.set(font_scale=0.5)
plot_2 = sns.barplot(y = data['State'], x = data['norm_registrations'])
plot_2.set(xlabel = 'State', ylabel= 'Number of EV registered', title = 'Electric Vehicle registrations by state 2008-2020')
plt.xticks(rotation = 90)

## Plot 4 Plotting education attainment and number of EVs registered

sns.set(style = 'white')
plt.figure(figsize=(6, 4), 
           dpi = 600)
plot_4 = sns.scatterplot(x = data['Percentage_pop_with_bachelors'], y = data['norm_registrations'])
plot_4.set(xlabel = 'Percentage of population with a Bachelors degree', ylabel = 'Normalized EV registrations', title = 'Relationship between Bachelors education level and EV registrations')

## Plot 5

#Adding a normalized GDP column

data['norm_GDP'] = (data['GDP']-data['GDP'].min())*100/(data['GDP'].max()-data['GDP'].min())
sns.set(style = 'white')
plt.figure(figsize=(6, 4), 
           dpi = 600)
plot_4 = sns.scatterplot(x = data['norm_GDP'], y = data['norm_registrations'])
plot_4.set(xlabel = 'Normalized GDP', ylabel = 'Normalized EV registrations', title = 'Relationship between normalized GDP and normalized EV registrations')















