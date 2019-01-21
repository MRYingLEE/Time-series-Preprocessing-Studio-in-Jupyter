The framework of the notebook is as the following:

# Time-series Data Preprocessing Studio

## Import some basic relevant packages
Ideally, one should set up their own virtual environment and determine the versions of each library that they are using.

## First Look at my data
- How many rows are in the dataset?
- How many columns are in this dataset?
- What data types are the columns?
- Is the data complete? Are there nulls? Do we have to infer values?
- What is the definition of these columns?
- What are some other caveats to the data?

## Profile my data (lengthy but very helpful)
Here, we need a library named as pandas_profiling (https://github.com/pandas-profiling/pandas-profiling), which creates HTML profiling reports from pandas DataFrame objects.

## Data Preview in an interactive grid
Qgrid is a scrollable grid widget that can be used to edit, sort, and filter DataFrames in Jupyter notebooks. It was developed for use in [Quantopian's hosted research environment](https://www.quantopian.com/notebooks/survey?utm_source=quantopian&amp;utm_medium=web&amp;utm_campaign=qgrid-demo-nb) and also was released as an [open source project on GitHub](https://github.com/quantopian/qgrid).

Please note: as of Jan 2019, Qgrid along with other Jupyter Widgets don't work in **Google Colab**.

## Preprocess my data (Deal with N/A)
The reference to deal with missing data can be found https://pandas.pydata.org/pandas-docs/stable/missing_data.html

## Preprocess my data (Others)

## Data Preview in Chart
- There may appear to be an overall increasing trend. 
- There may appear to be some differences in the variance over time. 
- There may be some seasonality (i.e., cycles) in the data.
- There may be some outliers.

## Deal with stationarity
Most time-series models assume that the underlying time-series data is **stationary**.  This assumption gives us some nice statistical properties that allows us to use various models for forecasting.

**Stationarity** is a statistical assumption that a time-series has:
*   **Constant mean**
*   **Constant variance**
*   **Autocovariance does not depend on time**

More simply put, if we are using past data to predict future data, we should assume that the data will follow the same general trends and patterns as in the past.  This general statement holds for most training data and modeling tasks.

**There are some good diagrams and explanations on stationarity [here](https://www.analyticsvidhya.com/blog/2015/12/complete-tutorial-time-series-modeling/) and [here](https://people.duke.edu/~rnau/411diff.htm).**

Sometimes we need to transform the data in order to make it stationary.  However, this  transformation then calls into question if this data is truly stationary and is suited to be modeled using these techniques.

We will use **Dickey-Fuller test** to check wheather the time series is stationary or not.

Reference: Test stationarity using moving average statistics and Dickey-Fuller test (https://www.analyticsvidhya.com/blog/2016/02/time-series-forecasting-codes-python/)

## Correct for stationarity

It is common for time series data to have to correct for non-stationarity. 

2 common reasons behind non-stationarity are:

1. **Trend** – mean is not constant over time.
2. **Seasonality** – variance is not constant over time.

There are ways to correct for trend and seasonality, to make the time series stationary.
**What happens if you do not correct for these things?**

Many things can happen, including:
- Variance can be mis-specified
- Model fit can be worse.  
- Not leveraging valuable time-dependent nature of the data.  

Here are some resources on the pitfalls of using traditional methods for time series analysis.  
[Quora link](https://www.quora.com/Why-cant-you-use-linear-regression-for-time-series-data)  
[Quora link](https://www.quora.com/Data-Science-Can-machine-learning-be-used-for-time-series-analysis)

### Eliminating trend and seasonality
*   **Transformation**
  *   *Examples.* Log, square root, etc.
*   **Smoothing**
  *  *Examples.* Weekly average, monthly average, rolling averages.
*   **Differencing**
  *  *Examples.* First-order differencing.
*   **Polynomial Fitting**
  *  *Examples.* Fit a regression model.
*   **Decomposition: trend, seasonality, residuals**

Here we use Decomposition.

# Additional data considerations before choosing a model
*   Whether or not to incorporate external data
*   Whether or not to keep as univariate or multivariate (i.e., which features and number of features)
*   Outlier detection and removal
*   Missing value imputation

# Go further -- Statistical models
*   **Ignore the time-series aspect completely and model using traditional statistical modeling toolbox.** 
  *   *Examples.* Regression-based models.  
*   **Univariate statistical time-series modeling.**
  *   *Examples.* Averaging and smoothing models, ARIMA models.
*   **Slight modifications to univariate statistical time-series modeling.**
  *    *Examples.* External regressors, multi-variate models.
*   **Additive or component models.**
  *  *Examples.* Facebook Prophet package.
*   **Structural time series modeling.**
  *    *Examples.* Bayesian structural time series modeling, hierarchical time series modeling.

  ## ARIMA models.
You may use ARIMA models when we know there is dependence between values and leverage that information to forecast.

**ARIMA = Auto-Regressive Integrated Moving Average**.   
Assumptions: The time-series is stationary.  
Depends on:
 1. Number of AR (Auto-Regressive) terms (p).
 2. Number of I (Integrated or Difference) terms (d).
 3. Number of MA (Moving Average) terms (q). 
  
 ## Facebook Prophet package.
[Facebook Prophet](https://facebook.github.io/prophet/), a tool that allows folks to forecast using additive or component models relatively easily.  It can also include things like:
* Day of week effects
* Day of year effects
* Holiday effects
* Trend trajectory
* Can do MCMC sampling

# Go further --Machine Learning.
*   **Ignore the time-series aspect completely and model using traditional machine learning modeling toolbox.** 
  *   *Examples.* Support Vector Machines (SVMs), Random Forest Regression, Gradient-Boosted Decision Trees (GBDTs).
*   **Hidden markov models (HMMs).**
*   **Other sequence-based models.**
*   **Gaussian processes (GPs).**
*   **Recurrent neural networks (RNNs).**

<span style="background-color: #FFFF00">Garbage in, Garbage out.</span>
In the data precessing, you should put a lot of effort on the knowing your data.

## Sklearn scalers
In the preprocessing, you may want to scale your original data.
sklearn provides the following scalers:
StandardScaler
MinMaxScaler
MaxAbsScaler
RobustScaler
PowerTransformer
QuantileTransformer (Gaussian output)
QuantileTransformer (uniform output)
Normalizer

Reference: Compare the effect of different scalers on data with outliers
https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html 

# Credit
Part code is modified from 
Applying statistical modeling and machine learning to perform time-series forecasting (https://goo.gl/r7CFcN). by Tamara Louie in PyData LA  October 2018 
