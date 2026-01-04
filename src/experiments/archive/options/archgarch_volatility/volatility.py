import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import seaborn as sns
import yfinance as yf
from sklearn.metrics import mean_squared_error
from arch import arch_model

#Volatility Prediction based on ARCH model
#Autoregressive Conditional Heteroskedasticity (ARCH) models variance as a function of the error terms (residuals)
# from a mean process. ARCH is a proper model in cases where the error variance in a time series follows an
# autoregressive model.
#We have talked about normal distribution before, which is symmetric and bell-shaped.
# But we have not discussed student-t and skewed distribution. Student-t distribution is a better choice when we try to
# model data with small samples and the population standard deviation is unknown.
#A distribution is said to be skewed when the data points cluster more toward one side of the scale than the other,
# creating a curve that is not symmetrical. Let's model the volatility using an ARCH model in Python.
# We will employ five different and arbitrarily selected stocks:

#T. Rowe Price Group (TROW)
#The Travelers Companies Inc. (TRV)
#Truist Financial (TFC)
#Wells Fargo (WFC)
#Zions Bancorp (ZION)

stocks = ['TROW','TRV','TFC','WFC','ZION']

#Using Yahoo Finance, we retrieve daily stock prices of our four stocks:
#The period that we cover is 2016/01/01 - 2020/12/01.
vol_data=yf.download(stocks,'2016-01-01', '2020-01-01')['Close']
vol_data.head()

#Now, the log-return of the stock price and the variables of interest is computed. Due to the convergence issue,
# we scale returns up by 100.
vol_return=np.log(vol_data/vol_data.shift(1)).dropna()*100
vol_return.head()

#We do the necessary cleaning and we are ready to apply the ARCH model. We will model the volatility with three
# different models and distributions. So, we have nine prediction results below.

#The following code both fits (results_arch_normal) the volatility model and performs the forecast
# (forecast_arch_normal) based on the fitted model.
#Split date is chosen to represent the last 100 days.

split_date='2019-08-09'

results_arch_normal=[]
forecast_arch_normal=[]
for i in vol_return.columns:
    for j in range(len(vol_return.columns)):
        results_arch_normal.append(arch_model((vol_return[i]),mean='Constant',vol='arch',dist='Normal').fit(disp='off'))
        forecast_arch_normal.append(results_arch_normal[j].forecast(start=split_date))

# The result of the ARCH model is shown—omega and alpha are constant terms and slope coefficient, respectively. Both of
# these estimated coefficients are statistically significant at a level of 1%.
results_arch_normal

window_size=5
rv_all=[]
for j in vol_return.columns:
    rv=(vol_return[j]).rolling(window_size).std()
    rv_all.append(rv)
rv_all=pd.DataFrame(rv_all)
rv_all=rv_all.T

#As the window size is defined as 5, we see values are missing and realize volatility starts at 2016-01-08.
#The question is: how can we know which model best fits our data? The Root Mean Squared Error is used to
# tackle this question.

len(forecast_arch_normal[0].variance[split_date:])
from sklearn.metrics import mean_squared_error

#The RMSE is calculated as 0.005916. It does not mean anything unless we compare it with other results.

rmse_all_arch_normal=[]
for i in rv_all.columns:
    for j in range(0,len(forecast_arch_normal)):
        rmse = np.sqrt(mean_squared_error(rv_all[i].iloc[-100:]/100, np.sqrt(forecast_arch_normal[j].variance[split_date:].mean(axis=1))/100))
        rmse_all_arch_normal.append(rmse)
print('The RMSE is: {:.6f}'.format(np.mean(rmse_all_arch_normal)))

#Though RMSE is a reliable way to measure the out-of-sample performance, visualization is another method to observe
# how good a prediction is. To do this, we complete the following analysis. The orange line in ARCH_normal.png plot
# represents the prediction and the blue lines denote realized volatility. The visualization result speaks for itself.

sns.set()
plt.figure(figsize=(20,10))
k=0
for i,j in zip(vol_return.columns,range(len(vol_return.columns))):
    k+=1
    plt.subplot(3,2,k)
    plt.tight_layout()
    plt.plot(rv_all[i]/100,label='Realized Volatility')
    plt.plot(np.sqrt(forecast_arch_normal[j].variance[split_date:].mean(axis=1))/100,label='ARCH Prediction with Normal Dist')
    plt.title(i, loc='left', fontsize=12)
    plt.legend(loc='best')
plt.show()

#After applying the ARCH model with normal distribution, we can now make volatility predictions with student's t
# distribution.
results_arch_student=[]
forecast_arch_student=[]
for i in vol_return.columns:
    for j in range(len(vol_return.columns)):
        results_arch_student.append(arch_model((vol_return[i]),mean='Constant',vol='arch',dist='studentst').fit(disp='off'))
        forecast_arch_student.append(results_arch_student[j].forecast(start=split_date))

#The RMSE (0.005992) that we get from ARCH with student's t distribution is larger than that (0.005916) of ARCH with
# normal distribution.

rmse_all_arch_student=[]
for i in rv_all.columns:
    for j in range(0,len(forecast_arch_student)):
        rmse = np.sqrt(mean_squared_error(rv_all[i][-100:]/100, np.sqrt(forecast_arch_student[j].variance[split_date:].mean(axis=1))/100))
        rmse_all_arch_student.append(rmse)
print('The RMSE is: {:.6f}'.format(np.mean(rmse_all_arch_student)))

sns.set()
plt.figure(figsize=(20,10))
k=0
for i,j in zip(vol_return.columns,range(len(vol_return.columns))):
    k+=1
    plt.subplot(3,2, j+1)
    plt.tight_layout()
    plt.plot(rv_all[i]/100,label='Realized Volatility')
    plt.plot(np.sqrt(forecast_arch_student[j].variance[split_date:].mean(axis=1))/100,label='ARCH Prediction with Student t Dist')
    plt.title(i, loc='left', fontsize=12)
    plt.legend(loc='best')
plt.show()

# As for the skewed distribution, we have an RMSE of 0.005916, meaning that an ARCH model with normal distribution is the best model, and an ARCH model with skewed distribution is the second best.

results_arch_skew=[]
forecast_arch_skew=[]
for i in vol_return.columns:
    for j in range(len(vol_return.columns)):
        results_arch_skew.append(arch_model((vol_return[i]),mean='Constant',vol='arch',dist='skewt').fit(disp='off'))
        forecast_arch_skew.append(results_arch_skew[j].forecast(start=split_date))

rmse_all_arch_skew=[]
for i in rv_all.columns:
    for j in range(0,len(forecast_arch_skew)):
        rmse = np.sqrt(mean_squared_error(rv_all[i][-100:]/100, np.sqrt(forecast_arch_skew[j].variance[split_date:].mean(axis=1))/100))
        rmse_all_arch_skew.append(rmse)
print('The RMSE is: {:.6f}'.format(np.mean(rmse_all_arch_skew)))

# The result is visualized in the ARCH_skewed.png file in the side panel.
sns.set()
plt.figure(figsize=(20,10))
k=0
for i,j in zip(vol_return.columns,range(len(vol_return.columns))):
    k+=1
    plt.subplot(3,2, k)
    plt.tight_layout()
    plt.plot(rv_all[i]/100,label='Realized Volatility')
    plt.plot(np.sqrt(forecast_arch_skew[j].variance[split_date:].mean(axis=1))/100,label='ARCH Prediction with Skewed Dist')
    plt.title(i, loc='left', fontsize=12)
    plt.legend(loc='best')
# Let's move forward with the GARCH model, which can extend the ARCH model so as to include past variance values.
# Volatility Prediction based on GARCH model
# Generalized Autoregressive Conditional Heteroskedasticity (GARCH) is an extension of ARMA and used to eliminate some
# of the drawbacks of ARCH. GARCH models are a best fit in cases where error terms follow the ARMA process.

# GARCH improves upon the ARCH model by adding lagged conditional variance into the equation. The mathematical equation
# is similar to ARCH except for one term, which is the past value of conditional volatility:

# Let's apply a GARCH model to the same data that we used before.

results_garch_normal=[]
forecast_garch_normal=[]
for i in vol_return.columns:
    for j in range(len(vol_return.columns)):
        results_garch_normal.append(arch_model((vol_return[i]),mean='Constant',vol='garch',dist='Normal').fit(disp='off'))
        forecast_garch_normal.append(results_garch_normal[j].forecast(start=split_date))

# Unlike in the ARCH model, we have one more variable $\beta$, attached to lagged values of conditional variance.
# The GARCH model result is exhibited below:
results_garch_normal
# The RMSE is 0.004662, which is lower than all RMSEs of the ARCH model. Therefore, a GARCH model with normal
# distribution fits better than the rest of the ARCH models.


rmse_all_garch_normal=[]
for i in rv_all.columns:
    for j in range(0,len(forecast_arch_skew)):
        rmse = np.sqrt(mean_squared_error(rv_all[i][-100:]/100, np.sqrt(forecast_garch_normal[j].variance[split_date:].mean(axis=1))/100))
        rmse_all_garch_normal.append(rmse)
print('The RMSE is: {:.6f}'.format(np.mean(rmse_all_garch_normal)))

# Notice the pattern of the orange line in the GARCH_normal.png plot—this confirms that a GARCH model fits
# better than the ARCH model.

sns.set()
plt.figure(figsize=(20,10))
k=0
for i,j in zip(vol_return.columns,range(len(vol_return.columns))):
    k+=1
    plt.subplot(3,2, k)
    plt.tight_layout()
    plt.plot(rv_all[i]/100,label='Realized Volatility')
    plt.plot(np.sqrt(forecast_garch_normal[j].variance[split_date:].mean(axis=1))/100,label='GARCH Prediction with Normal Dist')
    plt.title(i, loc='left', fontsize=12)
    plt.legend(loc='best')

results_garch_student=[]
forecast_garch_student=[]
for i in vol_return.columns:
    for j in range(len(vol_return.columns)):
        results_garch_student.append(arch_model((vol_return[i]),mean='Constant',vol='garch',dist='studentst').fit(disp='off'))
        forecast_garch_student.append(results_garch_student[j].forecast(start=split_date))
# The result of a GARCH model with student's t distribution is given below.

rmse_all_garch_student=[]
for i in rv_all.columns:
    for j in range(0,len(forecast_arch_skew)):
        rmse = np.sqrt(mean_squared_error(rv_all[i][-100:]/100, np.sqrt(forecast_garch_student[j].variance[split_date:].mean(axis=1))/100))
        rmse_all_garch_student.append(rmse)
print('The RMSE is: {:.6f}'.format(np.mean(rmse_all_garch_student)))
# The RMSE indicates that the GARCH with student's t distribution has lowered performance than the GARCH with normal distribution.

# TODO: value error occurred therefore deactivated
# sns.set()
# plt.figure(figsize=(20,10))
# k=0
# for i in vol_return.columns:
#     for j in range(len(vol_return.columns)):
#         k+=1
#         plt.subplot(3,2, k)
#         plt.tight_layout()
#         plt.plot(rv_all[i]/100,label='Realized Volatility')
#         plt.plot(np.sqrt(forecast_garch_student[j].variance[split_date:].mean(axis=1))/100,label='GARCH Prediction with Student Dist')
#         plt.title(i, loc='left', fontsize=12)
#         plt.legend(loc='best')

# The last GARCH application is one with skewed distribution and the result is provided here.

results_garch_skew=[]
forecast_garch_skew=[]
for i in vol_return.columns:
    for j in range(len(vol_return.columns)):
        results_garch_skew.append(arch_model((vol_return[i]),mean='Constant',vol='garch',dist='skewt').fit(disp='off'))
        forecast_garch_skew.append(results_garch_skew[j].forecast(start=split_date))

# The RMSE turns out similarly to the GARCH with normal distribution (at least at 6 digits).

rmse_all_garch_skew=[]
for i in rv_all.columns:
    for j in range(0,len(forecast_arch_skew)):
        rmse = np.sqrt(mean_squared_error(rv_all[i][-100:]/100, np.sqrt(forecast_garch_skew[j].variance[split_date:].mean(axis=1))/100))
        rmse_all_garch_skew.append(rmse)
print('The RMSE is: {:.6f}'.format(np.mean(rmse_all_garch_skew)))

sns.set()
plt.figure(figsize=(20,10))
k=0
for i,j in zip(vol_return.columns,range(len(vol_return.columns))):
    k+=1
    plt.subplot(3,2, k)
    plt.tight_layout()
    plt.plot(rv_all[i]/100,label='Realized Volatility')
    plt.plot(np.sqrt(forecast_garch_skew[j].variance[split_date:].mean(axis=1))/100,label='GARCH Prediction with Skewed Dist')
    plt.title(i, loc='left', fontsize=12)
    plt.legend(loc='best')

# EGARCH is the last model in this lab. Let's discover if it outperforms the GARCH and ARCH models.
#
# Volatility Prediction based on EGARCH model
# Exponential Generalized Autoregressive Conditional Heteroskedasticity (EGARCH) is a dynamic model that addresses the
# volatility clustering process, which is the tendency of large changes in prices of financial assets
# to cluster together.
#
# The first application is the EGARCH model with normal distribution. Let's look at the result:

results_egarch_normal=[]
forecast_egarch_normal=[]
for i in vol_return.columns:
    for j in range(len(vol_return.columns)):
        results_egarch_normal.append(arch_model((vol_return[i]),mean='Constant',vol='egarch',dist='normal').fit(disp='off'))
        forecast_egarch_normal.append(results_egarch_normal[j].forecast(start=split_date))

# Well, we model the data with EGARCH and store the result in results_egarch_normal. Let's see what it looks like:
rmse_all_egarch_normal=[]

# Now let's run the following code to calculate the RMSE metric:

for i in rv_all.columns:
    for j in range(0,len(forecast_arch_skew)):
        rmse = np.sqrt(mean_squared_error(rv_all[i][-100:]/100, np.sqrt(forecast_egarch_normal[j].variance[split_date:].mean(axis=1))/100))
        rmse_all_egarch_normal.append(rmse)
print('The RMSE is: {:.6f}'.format(np.mean(rmse_all_egarch_normal)))

# The RMSE is 0.004943; it is higher than GARCH and lower than ARCH models. Now, it is time to visualize the actual and predicted results so we can observe how accurate our prediction is.

#Even though the RMSE result seems to be promising, it is not the only way that we gauge the prediction result.
# Visualization provides another strong tool to see how a model works; ours can be found in the EGARCH_normal.png file.

sns.set()
plt.figure(figsize=(20,10))
k=0
for i,j in zip(vol_return.columns, range(len(vol_return.columns))):
        k+=1
        plt.subplot(3,2, k)
        plt.tight_layout()
        plt.plot(rv_all[i]/100,label='Realized Volatility')
        plt.plot(np.sqrt(forecast_egarch_normal[j].variance[split_date:].mean(axis=1))/100,label='EGARCH Prediction with Normal Dist')
        plt.title(i, loc='left', fontsize=12)
        plt.legend(loc='best')

results_egarch_student=[]
forecast_egarch_student=[]
for i in vol_return.columns:
    for j in range(len(vol_return.columns)):
        results_egarch_student.append(arch_model((vol_return[i]),mean='Constant',vol='egarch',dist='studentst').fit(disp='off'))
        forecast_egarch_student.append(results_egarch_student[j].forecast(start=split_date))


# After running EGARCH with the assumption of normally distributed error terms, the next step is to model the same data
# with the same model but with error terms following student's t-distribution.
rmse_all_egarch_student=[]
for i in rv_all.columns:
    for j in range(0,len(forecast_arch_skew)):
        rmse = np.sqrt(mean_squared_error(rv_all[i][-100:]/100, np.sqrt(forecast_egarch_student[j].variance[split_date:].mean(axis=1))/100))
        rmse_all_egarch_student.append(rmse)
print('The RMSE is: {:.6f}'.format(np.mean(rmse_all_egarch_student)))

sns.set()
plt.figure(figsize=(20,10))
k=0
for i,j in zip(vol_return.columns,range(len(vol_return.columns))):
    k+=1
    plt.subplot(3,2, k)
    plt.tight_layout()
    plt.plot(rv_all[i]/100,label='Realized Volatility')
    plt.plot(np.sqrt(forecast_egarch_student[j].variance[split_date:].mean(axis=1))/100,label='EGARCH Prediction with Student-t Dist')
    plt.title(i, loc='left', fontsize=12)
    plt.legend(loc='best')

results_egarch_skew=[]
forecast_egarch_skew=[]
for i in vol_return.columns:
    for j in range(len(vol_return.columns)):
        results_egarch_skew.append(arch_model((vol_return[i]),mean='Constant',vol='egarch',dist='skewt').fit(disp='off'))
        forecast_egarch_skew.append(results_egarch_skew[j].forecast(start=split_date))

rmse_all_egarch_skew=[]
for i,j in zip(rv_all.columns,range(0,len(forecast_arch_skew))):
    rmse = np.sqrt(mean_squared_error(rv_all[i][-100:]/100, np.sqrt(forecast_egarch_skew[j].variance[split_date:].mean(axis=1))/100))
    rmse_all_egarch_skew.append(rmse)
print('The RMSE is: {:.6f}'.format(np.mean(rmse_all_egarch_skew)))

sns.set()
plt.figure(figsize=(20,10))
k=0
for i,j in zip(vol_return.columns,range(len(vol_return.columns))):
    k+=1
    plt.subplot(3,2, k)
    plt.tight_layout()
    plt.plot(rv_all[i]/100,label='Realized Volatility')
    plt.plot(np.sqrt(forecast_egarch_skew[j].variance[split_date:].mean(axis=1))/100,label='EGARCH Prediction with Skewed Dist')
    plt.title(i, loc='left', fontsize=12)
    plt.legend(loc='best')

