

```python
# Load Libraries
import pandas as pd
from matplotlib import pyplot
import numpy as np
import datetime
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
```


```python
# Get Data
series = pd.read_csv('https://raw.githubusercontent.com/veeranalytics/Forecasting-ARIMA-Python/master/Product_Sales.csv')
```


```python
# Create a new column (Date) to capture date information in date format
series['Date'] = series['Month'].apply(lambda x: datetime.datetime.strptime(x, '%m/%d/%Y'))

# Delete original Date (Month) variable from dataframe
del series['Month']

# Set date as index
series.set_index('Date', inplace=True)
```


```python
# take a look at the data (first five obs)
series.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sales</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-01-01</th>
      <td>266.0</td>
    </tr>
    <tr>
      <th>2015-02-01</th>
      <td>145.9</td>
    </tr>
    <tr>
      <th>2015-03-01</th>
      <td>183.1</td>
    </tr>
    <tr>
      <th>2015-04-01</th>
      <td>119.3</td>
    </tr>
    <tr>
      <th>2015-05-01</th>
      <td>180.3</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Take a look at variable data types-- only one variable is there- Sales, Date variable has become index
# Converting Date variable to index helps in performing forecasting easier.
series.dtypes
```




    Sales    float64
    dtype: object




```python
# Show plot
series.plot()
pyplot.show()

## Findings:
## As the the plot-- there is an increasing linear trend as well as some seasonality component.
## The series has a clear trend-- this suggests that the time series is not stationary and 
## will require differencing to make it stationary, at least a difference order of 1.
## Will fit an ARIMA model.
```


![png](output_5_0.png)



```python
# Out of Sample Forecast Using ARIMA
# Divide series in to train ans test datasets
X = series.values
size = int(len(X) * 0.83)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
```


```python
# Checking Autocorrelation factor
# Autocorrelation: is the correlation between series values that are k intervals apart.
acf_plot = plot_acf(train)
```


![png](output_7_0.png)



```python
# Checking Partial Autocorrelation factor
# Autocorrelation: is the correlation between series values that are k intervals apart.
pacf_plot = plot_pacf(train)

## Findings:
## ACF: There is a positive correlation with the first 10 to 12 lags, perhaps significant for the first 3 lags only.
## This suggest of MA(3) model.
## PACF: The correlation is significant for the first 03 lags only, this suggests of AR(3) model.
## We will go with ARIMA(3,1,3) model.
```


![png](output_8_0.png)



```python
# Fit ARIMA model
model = ARIMA(train, order=(3,1,3))
model_fit = model.fit(disp=0)
print(model_fit.summary())
```

    C:\ProgramData\Anaconda3\lib\site-packages\statsmodels\tsa\kalmanf\kalmanfilter.py:646: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      if issubdtype(paramsdtype, float):
    

                                 ARIMA Model Results                              
    ==============================================================================
    Dep. Variable:                    D.y   No. Observations:                   28
    Model:                 ARIMA(3, 1, 3)   Log Likelihood                -150.598
    Method:                       css-mle   S.D. of innovations             46.893
    Date:                Sun, 20 May 2018   AIC                            317.197
    Time:                        15:19:21   BIC                            327.854
    Sample:                             1   HQIC                           320.455
                                                                                  
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          8.8651      3.336      2.658      0.015       2.327      15.403
    ar.L1.D.y     -0.1049      0.251     -0.417      0.681      -0.597       0.388
    ar.L2.D.y     -0.5019      0.157     -3.201      0.004      -0.809      -0.195
    ar.L3.D.y     -0.6263      0.187     -3.354      0.003      -0.992      -0.260
    ma.L1.D.y     -1.1763      0.340     -3.457      0.002      -1.843      -0.509
    ma.L2.D.y      1.2146      0.425      2.856      0.009       0.381       2.048
    ma.L3.D.y     -0.2257      0.426     -0.529      0.602      -1.061       0.610
                                        Roots                                    
    =============================================================================
                     Real           Imaginary           Modulus         Frequency
    -----------------------------------------------------------------------------
    AR.1            0.3231           -0.9993j            1.0502           -0.2002
    AR.2            0.3231           +0.9993j            1.0502            0.2002
    AR.3           -1.4477           -0.0000j            1.4477           -0.5000
    MA.1            0.4753           -0.8798j            1.0000           -0.1712
    MA.2            0.4753           +0.8798j            1.0000            0.1712
    MA.3            4.4304           -0.0000j            4.4304           -0.0000
    -----------------------------------------------------------------------------
    

    C:\ProgramData\Anaconda3\lib\site-packages\statsmodels\tsa\kalmanf\kalmanfilter.py:650: FutureWarning: Conversion of the second argument of issubdtype from `complex` to `np.complexfloating` is deprecated. In future, it will be treated as `np.complex128 == np.dtype(complex).type`.
      elif issubdtype(paramsdtype, complex):
    


```python
# plot residual errors
residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
pyplot.show()
residuals.plot(kind='kde')
pyplot.show()
print(residuals.describe())
```

    C:\ProgramData\Anaconda3\lib\site-packages\statsmodels\tsa\kalmanf\kalmanfilter.py:577: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      if issubdtype(paramsdtype, float):
    


![png](output_10_1.png)



![png](output_10_2.png)


                    0
    count   28.000000
    mean    -7.782527
    std     56.419481
    min   -128.965132
    25%    -42.381254
    50%     -4.960226
    75%     36.342816
    max     94.437553
    


```python
# Predict sales for next 07 months to comapre with test data
start_index = 30
end_index = 35
predictions = model_fit.predict(start=start_index, end=end_index, typ='levels')
```

    C:\ProgramData\Anaconda3\lib\site-packages\statsmodels\tsa\kalmanf\kalmanfilter.py:577: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      if issubdtype(paramsdtype, float):
    


```python
# plot
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()
```


![png](output_12_0.png)



```python
# Print Errors
error = ((predictions - test) ** 2).mean(axis=None)
print('Test MSE: %.3f' % error)
```

    Test MSE: 16662.500
    


```python
# Forecast sales for next 24 months
start_index = 36
end_index = 59
forecast = model_fit.predict(start=start_index, end=end_index, typ='levels')
```

    C:\ProgramData\Anaconda3\lib\site-packages\statsmodels\tsa\kalmanf\kalmanfilter.py:577: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      if issubdtype(paramsdtype, float):
    


```python
pyplot.plot(forecast, color='blue')
pyplot.show()
```


![png](output_15_0.png)

