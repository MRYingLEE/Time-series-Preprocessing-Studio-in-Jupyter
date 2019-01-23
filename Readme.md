
# Time-series Data Preprocessing Studio.

<span style="background-color: #FFFF00">Garbage in, Garbage out.</span> During data analysis, at first you should put a lot of effort know your data and preprocess it.

There is no ready solution for data preprocessing. So that this notebook was created for you to **INTERACTIVELY** do it. You have to modify the code snippets from time to time to suite **YOUR data and YOUR target**.

Ideally, one should set up their own virtual environment and determine the versions of each library that they are using.

You may use online Jupyter notebook service, but please note: as of Jan 2019, Jupyter Widgets don't work in **Google Colab**. So that the section code of <u>Data Preview in an interactive grid widget</u> NOT working normally. But other code still works.

## Import some basic relevant packages


```python
# import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from IPython.display import display, HTML, display_html
import seaborn as sns
import datetime

# set formatting
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
```

Please assign the correct CSV file name according to your need!


```python
csv_file='reviews.csv'
# read in CSV file data
df = pd.read_csv(csv_file)
```

# First Look at your data
- How many rows are in the dataset?
- How many columns are in this dataset?
- What data types are the columns?
- Is the data complete? Are there nulls? Do we have to infer values?
- What is the definition of these columns?
- What are some other caveats to the data?


```python
# look at data
display(HTML('<b>Head Data</b>'))
display(df.head())

# look a shape of data
display(HTML('<b>Dataframe Shape</b>'))
display(df.shape)

# look at data types. Ideally look at all rows. 
display(HTML('<b>Data Types</b>'))
display(df.iloc[:,:].dtypes)

# see if any columns have nulls. Ideally look at all rows. 
display(HTML('<b>Nulls</b>'))
display(df.iloc[:,:].isnull().any())

# display descriptive statistics
display(HTML('<b>Describe</b>'))
display(df.describe(percentiles=[0.25,0.5,0.75,0.85,0.95,0.99]))

# display duplicated index. You may check other columns too
display(HTML('<b>Duplicated</b>'))
df[df.index.duplicated(keep='last')]
#df[df.duplicated('column', keep='last')]
```


<b>Head Data</b>



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
      <th>listing_id</th>
      <th>date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>109</td>
      <td>2011-08-15</td>
    </tr>
    <tr>
      <th>1</th>
      <td>109</td>
      <td>2016-05-15</td>
    </tr>
    <tr>
      <th>2</th>
      <td>344</td>
      <td>2016-06-14</td>
    </tr>
    <tr>
      <th>3</th>
      <td>344</td>
      <td>2016-12-30</td>
    </tr>
    <tr>
      <th>4</th>
      <td>344</td>
      <td>2018-08-28</td>
    </tr>
  </tbody>
</table>
</div>



<b>Dataframe Shape</b>



    (1226674, 2)



<b>Data Types</b>



    listing_id     int64
    date          object
    dtype: object



<b>Nulls</b>



    listing_id    False
    date          False
    dtype: bool



<b>Describe</b>



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
      <th>listing_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1.226674e+06</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.175424e+07</td>
    </tr>
    <tr>
      <th>std</th>
      <td>7.968547e+06</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.090000e+02</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>4.403905e+06</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.173958e+07</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.829580e+07</td>
    </tr>
    <tr>
      <th>85%</th>
      <td>2.111148e+07</td>
    </tr>
    <tr>
      <th>95%</th>
      <td>2.475714e+07</td>
    </tr>
    <tr>
      <th>99%</th>
      <td>2.823253e+07</td>
    </tr>
    <tr>
      <th>max</th>
      <td>3.053957e+07</td>
    </tr>
  </tbody>
</table>
</div>



<b>Duplicated</b>





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
      <th>listing_id</th>
      <th>date</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



It is often to set correct index.


```python
# #An example to set to a correct index
# df['date']=pd.to_datetime(df['date'],infer_datetime_format=True)
# df.index=df.set_index(df['date'])
df.index
```




    RangeIndex(start=0, stop=1226674, step=1)



# Profile your data (lengthy but very helpful)
Here, we need a library named as pandas_profiling (https://github.com/pandas-profiling/pandas-profiling), which creates HTML profiling reports from pandas DataFrame objects.


```python
import pandas_profiling
pandas_profiling.ProfileReport(df)
```

    C:\Users\Sky\Anaconda3\envs\tfks\lib\site-packages\pandas_profiling\plot.py:15: UserWarning: matplotlib.pyplot as already been imported, this call will have no effect.
      matplotlib.use(BACKEND)
    




<meta charset="UTF-8">

<style>

        .variablerow {
            border: 1px solid #e1e1e8;
            border-top: hidden;
            padding-top: 2em;
            padding-bottom: 2em;
            padding-left: 1em;
            padding-right: 1em;
        }

        .headerrow {
            border: 1px solid #e1e1e8;
            background-color: #f5f5f5;
            padding: 2em;
        }
        .namecol {
            margin-top: -1em;
            overflow-x: auto;
        }

        .dl-horizontal dt {
            text-align: left;
            padding-right: 1em;
            white-space: normal;
        }

        .dl-horizontal dd {
            margin-left: 0;
        }

        .ignore {
            opacity: 0.4;
        }

        .container.pandas-profiling {
            max-width:975px;
        }

        .col-md-12 {
            padding-left: 2em;
        }

        .indent {
            margin-left: 1em;
        }

        .center-img {
            margin-left: auto !important;
            margin-right: auto !important;
            display: block;
        }

        /* Table example_values */
            table.example_values {
                border: 0;
            }

            .example_values th {
                border: 0;
                padding: 0 ;
                color: #555;
                font-weight: 600;
            }

            .example_values tr, .example_values td{
                border: 0;
                padding: 0;
                color: #555;
            }

        /* STATS */
            table.stats {
                border: 0;
            }

            .stats th {
                border: 0;
                padding: 0 2em 0 0;
                color: #555;
                font-weight: 600;
            }

            .stats tr {
                border: 0;
            }

            .stats td{
                color: #555;
                padding: 1px;
                border: 0;
            }


        /* Sample table */
            table.sample {
                border: 0;
                margin-bottom: 2em;
                margin-left:1em;
            }
            .sample tr {
                border:0;
            }
            .sample td, .sample th{
                padding: 0.5em;
                white-space: nowrap;
                border: none;

            }

            .sample thead {
                border-top: 0;
                border-bottom: 2px solid #ddd;
            }

            .sample td {
                width:100%;
            }


        /* There is no good solution available to make the divs equal height and then center ... */
            .histogram {
                margin-top: 3em;
            }
        /* Freq table */

            table.freq {
                margin-bottom: 2em;
                border: 0;
            }
            table.freq th, table.freq tr, table.freq td {
                border: 0;
                padding: 0;
            }

            .freq thead {
                font-weight: 600;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;

            }

            td.fillremaining{
                width:auto;
                max-width: none;
            }

            td.number, th.number {
                text-align:right ;
            }

        /* Freq mini */
            .freq.mini td{
                width: 50%;
                padding: 1px;
                font-size: 12px;

            }
            table.freq.mini {
                 width:100%;
            }
            .freq.mini th {
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
                max-width: 5em;
                font-weight: 400;
                text-align:right;
                padding-right: 0.5em;
            }

            .missing {
                color: #a94442;
            }
            .alert, .alert > th, .alert > td {
                color: #a94442;
            }


        /* Bars in tables */
            .freq .bar{
                float: left;
                width: 0;
                height: 100%;
                line-height: 20px;
                color: #fff;
                text-align: center;
                background-color: #337ab7;
                border-radius: 3px;
                margin-right: 4px;
            }
            .other .bar {
                background-color: #999;
            }
            .missing .bar{
                background-color: #a94442;
            }
            .tooltip-inner {
                width: 100%;
                white-space: nowrap;
                text-align:left;
            }

            .extrapadding{
                padding: 2em;
            }

            .pp-anchor{

            }

</style>

<div class="container pandas-profiling">
    <div class="row headerrow highlight">
        <h1>Overview</h1>
    </div>
    <div class="row variablerow">
    <div class="col-md-6 namecol">
        <p class="h4">Dataset info</p>
        <table class="stats" style="margin-left: 1em;">
            <tbody>
            <tr>
                <th>Number of variables</th>
                <td>2 </td>
            </tr>
            <tr>
                <th>Number of observations</th>
                <td>1226674 </td>
            </tr>
            <tr>
                <th>Total Missing (%)</th>
                <td>0.0% </td>
            </tr>
            <tr>
                <th>Total size in memory</th>
                <td>18.7 MiB </td>
            </tr>
            <tr>
                <th>Average record size in memory</th>
                <td>16.0 B </td>
            </tr>
            </tbody>
        </table>
    </div>
    <div class="col-md-6 namecol">
        <p class="h4">Variables types</p>
        <table class="stats" style="margin-left: 1em;">
            <tbody>
            <tr>
                <th>Numeric</th>
                <td>1 </td>
            </tr>
            <tr>
                <th>Categorical</th>
                <td>1 </td>
            </tr>
            <tr>
                <th>Boolean</th>
                <td>0 </td>
            </tr>
            <tr>
                <th>Date</th>
                <td>0 </td>
            </tr>
            <tr>
                <th>Text (Unique)</th>
                <td>0 </td>
            </tr>
            <tr>
                <th>Rejected</th>
                <td>0 </td>
            </tr>
            <tr>
                <th>Unsupported</th>
                <td>0 </td>
            </tr>
            </tbody>
        </table>
    </div>
    <div class="col-md-12" style="padding-left: 1em;">
        
        <p class="h4">Warnings</p>
        <ul class="list-unstyled"><li><a href="#pp_var_date"><code>date</code></a> has a high cardinality: 3144 distinct values  <span class="label label-warning">Warning</span></li><li>Dataset has 3791 duplicate rows <span class="label label-warning">Warning</span></li> </ul>
    </div>
</div>
    <div class="row headerrow highlight">
        <h1>Variables</h1>
    </div>
    <div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_date">date<br/>
            <small>Categorical</small>
        </p>
    </div><div class="col-md-3">
    <table class="stats ">
        <tr class="alert">
            <th>Distinct count</th>
            <td>3144</td>
        </tr>
        <tr>
            <th>Unique (%)</th>
            <td>0.3%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (n)</th>
            <td>0</td>
        </tr>
    </table>
</div>
<div class="col-md-6 collapse in" id="minifreqtable7361829457780347048">
    <table class="mini freq">
        <tr class="">
    <th>2018-11-12</th>
    <td>
        <div class="bar" style="width:1%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 0.3%">
            &nbsp;
        </div>
        3791
    </td>
</tr><tr class="">
    <th>2018-10-28</th>
    <td>
        <div class="bar" style="width:1%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 0.3%">
            &nbsp;
        </div>
        3575
    </td>
</tr><tr class="">
    <th>2018-10-21</th>
    <td>
        <div class="bar" style="width:1%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 0.3%">
            &nbsp;
        </div>
        3354
    </td>
</tr><tr class="other">
    <th>Other values (3141)</th>
    <td>
        <div class="bar" style="width:100%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 99.1%">
            1215954
        </div>
        
    </td>
</tr>
    </table>
</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#freqtable7361829457780347048, #minifreqtable7361829457780347048"
       aria-expanded="true" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="col-md-12 extrapadding collapse" id="freqtable7361829457780347048">
    
<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">2018-11-12</td>
        <td class="number">3791</td>
        <td class="number">0.3%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">2018-10-28</td>
        <td class="number">3575</td>
        <td class="number">0.3%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">2018-10-21</td>
        <td class="number">3354</td>
        <td class="number">0.3%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">2018-10-14</td>
        <td class="number">3315</td>
        <td class="number">0.3%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">2018-09-03</td>
        <td class="number">3294</td>
        <td class="number">0.3%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">2018-09-30</td>
        <td class="number">3127</td>
        <td class="number">0.3%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">2018-09-23</td>
        <td class="number">3099</td>
        <td class="number">0.3%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">2018-07-08</td>
        <td class="number">3097</td>
        <td class="number">0.3%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">2018-10-07</td>
        <td class="number">3026</td>
        <td class="number">0.2%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">2018-08-19</td>
        <td class="number">3018</td>
        <td class="number">0.2%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="other">
        <td class="fillremaining">Other values (3134)</td>
        <td class="number">1193978</td>
        <td class="number">97.3%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr>
</table>
</div>
</div><div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_listing_id">listing_id<br/>
            <small>Numeric</small>
        </p>
    </div><div class="col-md-6">
    <div class="row">
        <div class="col-sm-6">
            <table class="stats ">
                <tr>
                    <th>Distinct count</th>
                    <td>34327</td>
                </tr>
                <tr>
                    <th>Unique (%)</th>
                    <td>2.8%</td>
                </tr>
                <tr class="ignore">
                    <th>Missing (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Missing (n)</th>
                    <td>0</td>
                </tr>
                <tr class="ignore">
                    <th>Infinite (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Infinite (n)</th>
                    <td>0</td>
                </tr>
            </table>

        </div>
        <div class="col-sm-6">
            <table class="stats ">

                <tr>
                    <th>Mean</th>
                    <td>11754000</td>
                </tr>
                <tr>
                    <th>Minimum</th>
                    <td>109</td>
                </tr>
                <tr>
                    <th>Maximum</th>
                    <td>30539573</td>
                </tr>
                <tr class="ignore">
                    <th>Zeros (%)</th>
                    <td>0.0%</td>
                </tr>
            </table>
        </div>
    </div>
</div>
<div class="col-md-3 collapse in" id="minihistogram2359107911530742763">
    <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAABLCAYAAAA1fMjoAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4xLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvDW2N/gAAA7BJREFUeJzt2r1LK2kUgPEzZm2uJqI2cUjAxhSCSGCxs7JR0cLCSoj4P2ihC2KzhYWdZbC2MmK1lYHUVwnaBiL4BWKVUTBCcrbYVfDGe/ycmUSfH6QwmY93YB7emXEcVVUJ2NnZmSSTyaB3ixZ3enoqiUQi0H3%2BEeje/heNRkXkvwOOxWJhDAEtpFKpSDKZfDxvghRKII7jiIhILBZrCOTPv/558/Z%2B/j3%2BKeNCc3s4b4LUFvgegRZCIICBQAADgQAGAgEMBAIYCAQwEAhgIBDAQCCAgUAAA4EABgIBDAQCGEJ53f2z8Yo8/MIMAhgIBDB8iUusZvWeS7/34HLRP8wggOHbziBf6cb%2BrcfSrMfRjJhBAAOBAAYCAQwEAhi%2B7U36ewT12BbNg0C%2Boa/0BM9vXGIBBgIBDAQCGAgEMBAIYCAQwEAggIFAAAOBAAYCAQy8aoJX%2Ba6vpzCDAAYCAQwEAhgIBDAQCGDgKRZ88xWefDGDAAYCAQyhXGKpqoiIVCqVht9q1dugh4Mm8tw58fDdw3kTpFAC8TxPRESSyWQYu0cT69r4/W%2Be50lXV1dwgxERR0PIsl6vy8XFhUSjUXEcJ%2Bjdo8WoqnieJ67rSltbsHcFoQQCtApu0gEDgQAGAsGHFQoFmZ6eFtd1xXEc2d3dfdP6a2tr4jhOw6ejo8OnEb8egeDDbm9vZXh4WDY3N9%2B1/uLiolxeXj75DA4Oyuzs7CeP9B0U%2BEQiorlc7sl31WpVl5aW1HVd/fHjh46MjGg%2Bn//tNorFooqIFgoFn0f7Mt7Fgu8WFhbk5OREtre3xXVdyeVyMj4%2BLsfHxzIwMNCwfDablVQqJaOjoyGM9hdhF4qvRX6ZQUqlkjqOo%2Bfn50%2BWGxsb0%2BXl5Yb17%2B7utLu7W9fX130f62swg8BXh4eHoqqSSqWefF%2BtVqW3t7dh%2BZ2dHfE8TzKZTFBDNBEIfFWv1yUSicjBwYFEIpEnv3V2djYsn81mZWpqSuLxeFBDNBEIfJVOp6VWq8nV1dWL9xTlclny%2Bbzs7e0FNLqXEQg%2B7ObmRkql0uPf5XJZisWi9PT0SCqVkrm5OclkMrKxsSHpdFqur69lf39fhoaGZHJy8nG9ra0t6evrk4mJiTAO43lh3wSh9eXzeRWRhs/8/Lyqqt7f3%2Bvq6qr29/dre3u7xuNxnZmZ0aOjo8dt1Go1TSQSurKyEtJRPI%2BXFQED/0kHDAQCGAgEMBAIYCAQwEAggIFAAAOBAAYCAQwEAhj%2BBeN61QFvPyPjAAAAAElFTkSuQmCC">

</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#descriptives2359107911530742763,#minihistogram2359107911530742763"
       aria-expanded="false" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="row collapse col-md-12" id="descriptives2359107911530742763">
    <ul class="nav nav-tabs" role="tablist">
        <li role="presentation" class="active"><a href="#quantiles2359107911530742763"
                                                  aria-controls="quantiles2359107911530742763" role="tab"
                                                  data-toggle="tab">Statistics</a></li>
        <li role="presentation"><a href="#histogram2359107911530742763" aria-controls="histogram2359107911530742763"
                                   role="tab" data-toggle="tab">Histogram</a></li>
        <li role="presentation"><a href="#common2359107911530742763" aria-controls="common2359107911530742763"
                                   role="tab" data-toggle="tab">Common Values</a></li>
        <li role="presentation"><a href="#extreme2359107911530742763" aria-controls="extreme2359107911530742763"
                                   role="tab" data-toggle="tab">Extreme Values</a></li>

    </ul>

    <div class="tab-content">
        <div role="tabpanel" class="tab-pane active row" id="quantiles2359107911530742763">
            <div class="col-md-4 col-md-offset-1">
                <p class="h4">Quantile statistics</p>
                <table class="stats indent">
                    <tr>
                        <th>Minimum</th>
                        <td>109</td>
                    </tr>
                    <tr>
                        <th>5-th percentile</th>
                        <td>589620</td>
                    </tr>
                    <tr>
                        <th>Q1</th>
                        <td>4403900</td>
                    </tr>
                    <tr>
                        <th>Median</th>
                        <td>11740000</td>
                    </tr>
                    <tr>
                        <th>Q3</th>
                        <td>18296000</td>
                    </tr>
                    <tr>
                        <th>95-th percentile</th>
                        <td>24757000</td>
                    </tr>
                    <tr>
                        <th>Maximum</th>
                        <td>30539573</td>
                    </tr>
                    <tr>
                        <th>Range</th>
                        <td>30539464</td>
                    </tr>
                    <tr>
                        <th>Interquartile range</th>
                        <td>13892000</td>
                    </tr>
                </table>
            </div>
            <div class="col-md-4 col-md-offset-2">
                <p class="h4">Descriptive statistics</p>
                <table class="stats indent">
                    <tr>
                        <th>Standard deviation</th>
                        <td>7968500</td>
                    </tr>
                    <tr>
                        <th>Coef of variation</th>
                        <td>0.67793</td>
                    </tr>
                    <tr>
                        <th>Kurtosis</th>
                        <td>-1.122</td>
                    </tr>
                    <tr>
                        <th>Mean</th>
                        <td>11754000</td>
                    </tr>
                    <tr>
                        <th>MAD</th>
                        <td>6937600</td>
                    </tr>
                    <tr class="">
                        <th>Skewness</th>
                        <td>0.1897</td>
                    </tr>
                    <tr>
                        <th>Sum</th>
                        <td>14418620913238</td>
                    </tr>
                    <tr>
                        <th>Variance</th>
                        <td>63498000000000</td>
                    </tr>
                    <tr>
                        <th>Memory size</th>
                        <td>9.4 MiB</td>
                    </tr>
                </table>
            </div>
        </div>
        <div role="tabpanel" class="tab-pane col-md-8 col-md-offset-2" id="histogram2359107911530742763">
            <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAlgAAAGQCAYAAAByNR6YAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4xLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvDW2N/gAAIABJREFUeJzt3X9YlHW%2B//HXBAJGMpE/gEkz8qjJ6nopdATKTC38kbpWZ7XjRlKu6Wa6iB7zR3W0LX%2B0ZmfNsi1Lq%2B2EdRDXrtSkUtAjWBL4o8ysNHQFkVZBOQZI9/cPl/k6AsK4HxgGno/ruq/Lue/33POez043r/3c99xjsyzLEgAAAIy5ytMNAAAANDcELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwzNfTDbQEP//8s44fP642bdrIZrN5uh0AAFoEy7J05swZORwOXXVV484pEbAawfHjx9WpUydPtwEAQIt09OhRdezYsVFfk4DVCNq0aSPpwv/AQUFBHu4GAICWoaSkRJ06dXL%2BHW5MBKxGUHVaMCgoiIAFAEAj88TlOVzkDgAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADD%2BLFnLxc1b7OnW6i33c8O9XQLAAA0CmawAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhHglYixYt0i233KI2bdqoQ4cOGj16tA4ePOhSU1ZWpqlTp6pdu3YKDAzUqFGjdOzYMZeavLw8jRw5UoGBgWrXrp2mTZum8vJyl5r09HRFRkYqICBAN910k1555ZVq/bz88ssKDw9XQECAIiMjtX37drd7AQAAqOKRgJWenq4pU6YoKytLaWlpOn/%2BvOLi4lRaWuqsSUxMVGpqqpKTk7Vjxw6dPXtWI0aMUGVlpSSpsrJSd999t0pLS7Vjxw4lJycrJSVFM2bMcO7j8OHDGj58uPr376%2BcnBzNnTtX06ZNU0pKirNm7dq1SkxM1Lx585STk6P%2B/ftr2LBhysvLq3cvAAAAF7NZlmV5uomTJ0%2BqQ4cOSk9P1%2B23367i4mK1b99eb7/9tsaOHStJOn78uDp16qSNGzdqyJAh2rRpk0aMGKGjR4/K4XBIkpKTk5WQkKDCwkIFBQXp8ccf14YNG3TgwAHna02ePFl79uxRZmamJKlfv37q27evVq5c6azp0aOHRo8erUWLFtWrl7qUlJTIbreruLhYQUFBxsZNkqLmbTa6v4a0%2B9mhnm4BANCCNOTf37o0iWuwiouLJUnXXXedJCk7O1sVFRWKi4tz1jgcDvXs2VM7d%2B6UJGVmZqpnz57OcCVJQ4YMUVlZmbKzs501F%2B%2Bjqmb37t2qqKhQeXm5srOzq9XExcU5X6c%2BvVyqrKxMJSUlLgsAAGg5PB6wLMtSUlKSbrvtNvXs2VOSVFBQID8/PwUHB7vUhoSEqKCgwFkTEhLisj04OFh%2Bfn6XrQkJCdH58%2BdVVFSkoqIiVVZW1lhz8T7q6uVSixYtkt1udy6dOnVyZ0gAAICX83jAeuyxx7R37169%2B%2B67ddZaliWbzeZ8fPG/61tTdUa0rpqa9l3fmjlz5qi4uNi5HD169LL7AgAAzYtHA9bUqVO1YcMGbd26VR07dnSuDw0NVXl5uU6dOuVSX1hY6JxtCg0NrTaDdOrUKVVUVFy2prCwUL6%2Bvmrbtq3atWsnHx%2BfGmsu3kddvVzK399fQUFBLgsAAGg5PBKwLMvSY489pnXr1unTTz9VeHi4y/bIyEi1atVKaWlpznX5%2Bfnav3%2B/YmNjJUkxMTHav3%2B/8vPznTVbtmyRv7%2B/IiMjnTUX76OqJioqSq1atZKfn58iIyOr1aSlpTlfpz69AAAAXMzXEy86ZcoU/fd//7f%2B%2Bte/qk2bNs4ZJLvdrtatW8tut2vChAmaMWOG2rZtq%2Buuu04zZ85Ur169dOedd0q6cCF6RESE4uPj9cc//lF///vfNXPmTE2cONE5YzR58mStWLFCSUlJmjhxojIzM/X666%2B7nI5MSkpSfHy8oqKiFBMTo1dffVV5eXmaPHmys6e6egEAALiYRwJW1S0R7rjjDpf1q1evVkJCgiTphRdekK%2Bvr8aMGaNz585p8ODBWrNmjXx8fCRJPj4%2B%2BvDDD/Xoo4/q1ltvVevWrTVu3DgtXbrUub/w8HBt3LhR06dP10svvSSHw6Hly5frvvvuc9aMHTtWP/74o55%2B%2Bmnl5%2BerZ8%2Be2rhxozp37uysqasXAACAizWJ%2B2A1d9wH6wLugwUAaEwt/j5YAAAAzQkBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMMzX0w2g5Yiat9nTLbhl97NDPd0CAMBLMYMFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhnkkYGVkZGjkyJFyOByy2Wxav369y/aEhATZbDaXJTo62qWmrKxMU6dOVbt27RQYGKhRo0bp2LFjLjV5eXkaOXKkAgMD1a5dO02bNk3l5eUuNenp6YqMjFRAQIBuuukmvfLKK9X6ffnllxUeHq6AgABFRkZq%2B/bthkYCAAA0Rx4JWKWlperdu7dWrFhRa83QoUOVn5/vXDZu3OiyPTExUampqUpOTtaOHTt09uxZjRgxQpWVlZKkyspK3X333SotLdWOHTuUnJyslJQUzZgxw7mPw4cPa/jw4erfv79ycnI0d%2B5cTZs2TSkpKc6atWvXKjExUfPmzVNOTo769%2B%2BvYcOGKS8vz/CoAACA5sJmWZbl0QZsNqWmpmr06NHOdQkJCTp9%2BnS1ma0qxcXFat%2B%2Bvd5%2B%2B22NHTtWknT8%2BHF16tRJGzdu1JAhQ7Rp0yaNGDFCR48elcPhkCQlJycrISFBhYWFCgoK0uOPP64NGzbowIEDzn1PnjxZe/bsUWZmpiSpX79%2B6tu3r1auXOms6dGjh0aPHq1FixbV6z2WlJTIbreruLhYQUFB7g1QHbzt3lLehPtgAYB3a8i/v3Vpstdgbdu2TR06dFC3bt00ceJEFRYWOrdlZ2eroqJCcXFxznUOh0M9e/bUzp07JUmZmZnq2bOnM1xJ0pAhQ1RWVqbs7GxnzcX7qKrZvXu3KioqVF5eruzs7Go1cXFxztcBAAC4VJO8k/uwYcP061//Wp07d9bhw4f15JNPatCgQcrOzpa/v78KCgrk5%2Ben4OBgl%2BeFhISooKBAklRQUKCQkBCX7cHBwfLz87tsTUhIiM6fP6%2BioiJZlqXKysoaa6r2UZOysjKVlZU5H5eUlLg/CAAAwGs1yYBVddpPknr27KmoqCh17txZH374oe69995an2dZlmw2m/Pxxf%2Bub03VGVObzeby78vt41KLFi3SggULat0OAACatyZ7ivBiYWFh6ty5sw4dOiRJCg0NVXl5uU6dOuVSV1hY6JxtCg0NrTbLdOrUKVVUVFy2prCwUL6%2Bvmrbtq3atWsnHx%2BfGmsundW62Jw5c1RcXOxcjh49emVvHAAAeCW3AtZf/vIX/fTTTw3VS61%2B/PFHHT16VGFhYZKkyMhItWrVSmlpac6a/Px87d%2B/X7GxsZKkmJgY7d%2B/X/n5%2Bc6aLVu2yN/fX5GRkc6ai/dRVRMVFaVWrVrJz89PkZGR1WrS0tKcr1MTf39/BQUFuSwAAKDlcCtgJSUlKTQ0VJMmTdJnn312xS969uxZ5ebmKjc3V9KF2yXk5uYqLy9PZ8%2Be1cyZM5WZmakjR45o27ZtGjlypNq1a6d77rlHkmS32zVhwgTNmDFDn3zyiXJycvTAAw%2BoV69euvPOOyVduBA9IiJC8fHxysnJ0SeffKKZM2dq4sSJzsAzefJk/fDDD0pKStKBAwf0xhtv6PXXX9fMmTNd3vOqVav0xhtv6MCBA5o%2Bfbry8vI0efLkK37/AACgeXPrGqzjx49rw4YNWrNmjW677TZ17dpVDz/8sB588EG1b9%2B%2B3vvZvXu3Bg4c6HyclJQkSRo/frxWrlypffv26a233tLp06cVFhamgQMHau3atWrTpo3zOS%2B88IJ8fX01ZswYnTt3ToMHD9aaNWvk4%2BMjSfLx8dGHH36oRx99VLfeeqtat26tcePGaenSpc59hIeHa%2BPGjZo%2BfbpeeuklORwOLV%2B%2BXPfdd5%2BzZuzYsfrxxx/19NNPKz8/Xz179tTGjRvVuXNnd4YOAAC0IFd8H6yCggK99dZbevPNN/Xtt9/q7rvv1oQJEzR8%2BPDLXgDeEnEfLO/EfbAAwLt55X2wQkNDNXjwYN1xxx2y2WzavXu3xo0bp65du/JTMgAAoEVz%2BzYNRUVF%2Bstf/qLVq1fr4MGDGjlypNavX68hQ4aotLRUTzzxhB588EEdPny4IfoF0Aww89qwmH0FPM%2BtgHXPPfdo48aNCg8P129/%2B1uNHz/e5dqra665RrNmzdLy5cuNNwoAAOAt3ApYQUFB%2Bvjjj9W/f/9aa8LCwpz3qwIAAGiJ3ApYb775Zp01NptNXbp0ueKGAAAAvJ1bF7lPnz5dK1asqLb%2BpZde0owZM4w1BQAA4M3cCljvv/%2B%2BoqOjq62PiYnR2rVrjTUFAADgzdwKWEVFRQoODq62PigoSEVFRcaaAgAA8GZuBawuXbroo48%2Bqrb%2Bo48%2BUnh4uLGmAAAAvJlbF7knJiYqMTFRP/74owYNGiRJ%2BuSTT/Tcc8%2B5/AQNAABAS%2BZWwJo4caJ%2B%2BuknLVy4UP/5n/8pSerYsaOWL1%2Buhx9%2BuEEaBAAA8DZu38l96tSpmjp1qvLz89W6dWtde%2B21DdEXAACA13I7YFUJCwsz2QcAAECz4dZF7idPntRDDz2kG264QQEBAfLz83NZAAAA4OYMVkJCgr777jv9x3/8h8LCwmSz2RqqLwDAFfKmH9Pmh6nRXLkVsDIyMpSRkaE%2Bffo0VD9Ak%2BFNf6Qk/lABQFPi1inCjh07MmsFAABQB7cC1gsvvKA5c%2Bbo2LFjDdUPAACA13PrFGF8fLzOnDmjzp07KygoSK1atXLZXlhYaLQ5AAAAb%2BRWwFq8eHFD9QEAANBsuBWwJkyY0FB9AAAANBtuXYMlSUeOHNH8%2BfMVHx/vPCW4ZcsWHThwwHhzAAAA3sitgLV9%2B3b94he/UHp6ut577z2dPXtWkvTFF1/oqaeeapAGAQAAvI1bAevxxx/X/PnztXXrVpc7tw8aNEhZWVnGmwMAAPBGbgWsvXv36t/%2B7d%2Bqre/QoYNOnjxprCkAAABv5lbAuvbaa1VQUFBtfW5urq6//npjTQEAAHgztwLW/fffr9mzZ%2BvkyZPOO7rv2rVLM2fO1AMPPNAgDQIAAHgbtwLWwoULFRoaqrCwMJ09e1YRERGKjY3VLbfcoieffLKhegQAAPAqbt0Hy8/PT2vXrtU333yjL774Qj///LP69u2rm2%2B%2BuaH6AwAA8DpuBawq3bp1U7du3Uz3AgAA0Cy4FbAeeeSRy25/9dVX/6lmAAAAmgO3AlZ%2Bfr7L44qKCn355Zc6c%2BaMbr/9dqONAQAAeCu3AtYHH3xQbd358%2Bf1u9/9Tj169DDWFAAAgDdz%2B7cIL%2BXr66uZM2fqj3/8o4l%2BAAAAvN4/HbAk6fvvv1dFRYWJXQEAAHg9t04Rzpo1y%2BWxZVnKz8/Xhg0b9Jvf/MZoYwAAAN7KrYCVmZnp8viqq65S%2B/bttXjxYk2cONFoYwAAAN7KrYC1ffv2huoDAACg2TByDRYAAAD%2BP7dmsG655RbnjzzX5bPPPruihgAAALydWwFr4MCB%2BvOf/6xu3bopJiZGkpSVlaWDBw9q0qRJ8vf3b5AmAQAAvIlbAev06dOaMmWKFi5c6LJ%2B3rx5OnHihFatWmW0OQAAAG/k1jVY7733nh566KFq6xMSEvT%2B%2B%2B8bawoAAMCbuRWw/P39tXPnzmrrd%2B7cyelBAACAf3DrFOG0adM0efJk5eTkKDo6WtKFa7Bee%2B01zZ07t0EaBAAA8DZuBax58%2BYpPDxcf/rTn/TGG29Iknr06KHXXntN48aNa5AGAQAAvI1bAUuSxo0bR5gCAAC4DLdvNFpSUqI1a9boqaee0qlTpyRJe/bsUX5%2BvvHmAAAAvJFbM1j79%2B/XnXfeqauvvlpHjx5VQkKCgoOD9d577%2BnYsWN68803G6pPAHWImrfZ0y0AAP7BrRms6dOna9y4cfruu%2B8UEBDgXH/33XcrIyPDeHMAAADeyK0ZrM8//1wrV66s9nM5119/PacIAQAA/sGtGSw/Pz%2BdPXu22vpDhw6pXbt2xpoCAADwZm4FrFGjRukPf/iDzp8/L0my2Wz629/%2BptmzZ%2Bvee%2B9tkAYBAAC8jVsB6/nnn9fx48cVGhqqc%2BfOadCgQbrpppsUEBBQ7fcJAQAAWiq3rsGy2%2B3auXOn0tLS9MUXX%2Bjnn39W3759NWTIkGrXZQEAALRU9Z7Bqqio0F133aVvv/1WcXFxmj17tubOnauhQ4e6Ha4yMjI0cuRIORwO2Ww2rV%2B/3mW7ZVmaP3%2B%2BHA6HWrdurTvuuENffvmlS82pU6cUHx8vu90uu92u%2BPh4nT592qVm3759GjBggFq3bq3rr79eTz/9tCzLcqlJSUlRRESE/P39FRERodTUVLd7AQAAuFi9Z7BatWqlnJwcIzNVpaWl6t27tx566CHdd9991bY/99xzWrZsmdasWaNu3brpmWee0V133aWDBw%2BqTZs2ki7cUf7YsWPavPnCvX8eeeQRxcfH64MPPpB04Yaod911lwYOHKjPP/9c33zzjRISEhQYGKgZM2ZIkjIzMzV27Fj94Q9/0D333KPU1FSNGTNGO3bsUL9%2B/erdCwDgynjb/dt2PzvU0y3AS9isS6d0LiMxMVGBgYF69tlnzTVgsyk1NVWjR4%2BWdGHGyOFwKDExUY8//rgkqaysTCEhIVqyZIkmTZqkAwcOKCIiQllZWc4glJWVpZiYGH399dfq3r27Vq5cqTlz5ujEiRPy9/eXJC1evFgvvviijh07JpvNprFjx6qkpESbNm1y9jN06FAFBwfr3XffrVcv9VFSUiK73a7i4mIFBQUZGzvJ%2Bw5OAODNCFjepSH//tbF7Z/KWbFihfr166cpU6Zo1qxZLosJhw8fVkFBgeLi4pzr/P39NWDAAO3cuVPShZknu93uDFeSFB0d7bxGrKpmwIABznAlSUOGDNHx48d15MgRZ83Fr1NVU7WP%2BvRSk7KyMpWUlLgsAACg5XDrIvfs7Gz98pe/lCTt3bvXZZupi9wLCgokSSEhIS7rQ0JC9MMPPzhrOnToUO25HTp0cD6/oKBAN954Y7V9VG0LDw9XQUFBja9z8T7q6qUmixYt0oIFCy77PgEAQPNVr4D1/fffKzw8XNu3b2/ofpwuDWyWZbmsqynQ1VVTdTa0rppL19Wn5mJz5sxRUlKS83FJSYk6depUaz0AAGhe6nWKsGvXrjp58qTz8dixY3XixIkGaSg0NFTS/589qlJYWOicSQoNDa3x9U%2BePOlSU9M%2BJNVZc/H2unqpib%2B/v4KCglwWAADQctQrYF16HfzGjRtVWlraIA2Fh4crNDRUaWlpznXl5eVKT09XbGysJCkmJkbFxcX67LPPnDW7du1ScXGxS01GRobKy8udNVu2bJHD4XCeOoyJiXF5naqaqn3UpxcAAIBLuX2Ruwlnz55Vbm6ucnNzJV24mDw3N1d5eXmy2WxKTEzUwoULlZqaqv379yshIUFXX321xo0bJ0nq0aOHhg4dqokTJyorK0tZWVmaOHGiRowYoe7du0u6cBsHf39/JSQkaP/%2B/UpNTdXChQuVlJTkPL33%2B9//Xlu2bNGSJUv09ddfa8mSJfr444%2BVmJgoSfXqBQAA4FL1ugbLZrPVeV2SO3bv3q2BAwc6H1ddrzR%2B/HitWbNGs2bN0rlz5/Too4/q1KlT6tevn7Zs2eJy36l33nlH06ZNc37Db9SoUVqxYoVzu91uV1pamqZMmaKoqCgFBwcrKSnJ5dqo2NhYJScn64knntCTTz6pLl26aO3atS7fTqxPLwAAABer132wrrrqKg0bNsx5y4MPPvhAgwYNUmBgoEvdunXrGqZLL8d9sACgeeA%2BWN7Fk/fBqtcM1vjx410eP/DAAw3SDAAAQHNQr4C1evXqhu4DAACg2fDIRe4AAADNGQELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAw5pkwJo/f75sNpvLEhoa6txuWZbmz58vh8Oh1q1b64477tCXX37pso9Tp04pPj5edrtddrtd8fHxOn36tEvNvn37NGDAALVu3VrXX3%2B9nn76aVmW5VKTkpKiiIgI%2Bfv7KyIiQqmpqQ33xgEAQLPQJAOWJP3iF79Qfn6%2Bc9m3b59z23PPPadly5ZpxYoV%2BvzzzxUaGqq77rpLZ86ccdaMGzdOubm52rx5szZv3qzc3FzFx8c7t5eUlOiuu%2B6Sw%2BHQ559/rhdffFFLly7VsmXLnDWZmZkaO3as4uPjtWfPHsXHx2vMmDHatWtX4wwCAADwSjbr0imbJmD%2B/Plav369cnNzq22zLEsOh0OJiYl6/PHHJUllZWUKCQnRkiVLNGnSJB04cEARERHKyspSv379JElZWVmKiYnR119/re7du2vlypWaM2eOTpw4IX9/f0nS4sWL9eKLL%2BrYsWOy2WwaO3asSkpKtGnTJufrDx06VMHBwXr33Xfr/X5KSkpkt9tVXFysoKCgf2Zoqomat9no/gAAtdv97FBPtwA3NOTf37o02RmsQ4cOyeFwKDw8XPfff7%2B%2B//57SdLhw4dVUFCguLg4Z62/v78GDBignTt3Srow82S3253hSpKio6Nlt9tdagYMGOAMV5I0ZMgQHT9%2BXEeOHHHWXPw6VTVV%2BwAAAKhJkwxY/fr101tvvaWPPvpIr732mgoKChQbG6sff/xRBQUFkqSQkBCX54SEhDi3FRQUqEOHDtX226FDB5eamvZRte1yNVXba1NWVqaSkhKXBQAAtBy%2Bnm6gJsOGDXP%2Bu1evXoqJiVGXLl305ptvKjo6WpJks9lcnmNZlsu6S7fXp6bqbGldNTXt%2B2KLFi3SggULLlsDAACaryY5g3WpwMBA9erVS4cOHXJ%2Bm/DSWaTCwkLnbFNoaKhOnDhRbT8nT550qalpH5LqrLl0VutSc%2BbMUXFxsXM5evRofd8qAABoBrwiYJWVlenAgQMKCwtTeHi4QkNDlZaW5txeXl6u9PR0xcbGSpJiYmJUXFyszz77zFmza9cuFRcXu9RkZGSovLzcWbNlyxY5HA7deOONzpqLX6eqpmoftfH391dQUJDLAgAAWo4meYpw5syZGjlypG644QYVFhbqmWeeUUlJicaPHy%2BbzabExEQtXLhQXbt2VdeuXbVw4UJdffXVGjdunCSpR48eGjp0qCZOnKg///nPkqRHHnlEI0aMUPfu3SVduI3DggULlJCQoLlz5%2BrQoUNauHChnnrqKecpwN///ve6/fbbtWTJEv3qV7/SX//6V3388cfasWOHZwYGAOBR3vbNbb716DlNMmAdO3ZM//7v/66ioiK1b99e0dHRysrKUufOnSVJs2bN0rlz5/Too4/q1KlT6tevn7Zs2aI2bdo49/HOO%2B9o2rRpzm8Bjho1SitWrHBut9vtSktL05QpUxQVFaXg4GAlJSUpKSnJWRMbG6vk5GQ98cQTevLJJ9WlSxetXbvW5duJAAAAl2qS98FqbrgPFgDAE1r6DBb3wQIAAGhGCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMO%2BSQmzAAAMF0lEQVQIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMF9PNwAAABpG1LzNnm6h3nY/O9TTLRjFDBYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAaueXn75ZYWHhysgIECRkZHavn27p1sCAABNFAGrHtauXavExETNmzdPOTk56t%2B/v4YNG6a8vDxPtwYAAJogAlY9LFu2TBMmTNBvf/tb9ejRQ//1X/%2BlTp06aeXKlZ5uDQAANEG%2Bnm6gqSsvL1d2drZmz57tsj4uLk47d%2B6s8TllZWUqKytzPi4uLpYklZSUGO%2BvsqzU%2BD4BAGhsDfE3smqflmUZ33ddCFh1KCoqUmVlpUJCQlzWh4SEqKCgoMbnLFq0SAsWLKi2vlOnTg3SIwAA3s7%2BfMPt%2B8yZM7Lb7Q33AjUgYNWTzWZzeWxZVrV1VebMmaOkpCTn459//ll///vf1bZt21qfcyVKSkrUqVMnHT16VEFBQcb22xwwNpfH%2BNSOsbk8xqd2jE3tPDU2lmXpzJkzcjgcjfaaVQhYdWjXrp18fHyqzVYVFhZWm9Wq4u/vL39/f5d11157bYP1GBQUxH/MtWBsLo/xqR1jc3mMT%2B0Ym9p5Ymwae%2BaqChe518HPz0%2BRkZFKS0tzWZ%2BWlqbY2FgPdQUAAJoyZrDqISkpSfHx8YqKilJMTIxeffVV5eXlafLkyZ5uDQAANEE%2B8%2BfPn%2B/pJpq6nj17qm3btlq4cKGWLl2qc%2BfO6e2331bv3r093Zp8fHx0xx13yNeXrHwpxubyGJ/aMTaXx/jUjrGpXUsbG5vlie8uAgAANGNcgwUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFhN3Msvv6zw8HAFBAQoMjJS27dvv2x9SkqKIiIi5O/vr4iICKWmpjZSp43PnbFZs2aNbDZbteWnn35qxI4bR0ZGhkaOHCmHwyGbzab169fX%2BZz09HRFRkYqICBAN910k1555ZVG6NQz3B2fbdu21fjZ%2Bfrrrxup48azaNEi3XLLLWrTpo06dOig0aNH6%2BDBg3U%2BryUcd65kbFrScWflypX65S9/6byRaExMjDZt2nTZ5zT3zw0Bqwlbu3atEhMTNW/ePOXk5Kh///4aNmyY8vLyaqzPzMzU2LFjFR8frz179ig%2BPl5jxozRrl27Grnzhufu2EgX7iCcn5/vsgQEBDRi142jtLRUvXv31ooVK%2BpVf/jwYQ0fPlz9%2B/dXTk6O5s6dq2nTpiklJaWBO/UMd8enysGDB10%2BO127dm2gDj0nPT1dU6ZMUVZWltLS0nT%2B/HnFxcWptLT2H5VvKcedKxkbqeUcdzp27KjFixdr9%2B7d2r17twYNGqRf/epX%2BvLLL2usbxGfGwtN1r/%2B679akydPdll38803W7Nnz66xfsyYMdbQoUNd1g0ZMsS6//77G6xHT3F3bFavXm3Z7fbGaK1JkWSlpqZetmbWrFnWzTff7LJu0qRJVnR0dEO21iTUZ3y2bt1qSbJOnTrVSF01HYWFhZYkKz09vdaalnTcuVh9xqalHneqBAcHW6tWrapxW0v43DCD1USVl5crOztbcXFxLuvj4uK0c%2BfOGp%2BTmZlZrX7IkCG11nurKxkbSTp79qw6d%2B6sjh07asSIEcrJyWnoVr1CbZ%2Bb3bt3q6KiwkNdNT19%2BvRRWFiYBg8erK1bt3q6nUZRXFwsSbruuutqrWkpx51L1WdspJZ53KmsrFRycrJKS0sVExNTY01L%2BNwQsJqooqIiVVZWVvtB6ZCQkGo/PF2loKDArXpvdSVjc/PNN2vNmjXasGGD3n33XQUEBOjWW2/VoUOHGqPlJq22z8358%2BdVVFTkoa6ajrCwML366qtKSUnRunXr1L17dw0ePFgZGRmebq1BWZalpKQk3XbbberZs2etdS3luHOx%2Bo5NSzvu7Nu3T9dcc438/f01efJkpaamKiIiosbalvC5aRn3q/diNpvN5bFlWdXW/TP13syd9xodHa3o6Gjn41tvvVV9%2B/bViy%2B%2BqOXLlzdon96gprGsaX1L1L17d3Xv3t35OCYmRkePHtXSpUt1%2B%2B23e7CzhvXYY49p79692rFjR521Lem4I9V/bFracad79%2B7Kzc3V6dOnlZKSovHjxys9Pb3WkNXcPzfMYDVR7dq1k4%2BPT7U0X1hYWC31VwkNDXWr3ltdydhc6qqrrtItt9zSbP%2BfpDtq%2B9z4%2Bvqqbdu2HuqqaYuOjm7Wn52pU6dqw4YN2rp1qzp27HjZ2pZy3Knizthcqrkfd/z8/PQv//IvioqK0qJFi9S7d2/96U9/qrG2JXxuCFhNlJ%2BfnyIjI5WWluayPi0tTbGxsTU%2BJyYmplr9li1baq33VlcyNpeyLEu5ubkKCwtriBa9Sm2fm6ioKLVq1cpDXTVtOTk5zfKzY1mWHnvsMa1bt06ffvqpwsPD63xOSznuXMnY1LSPlnTcsSxLZWVlNW5rEZ8bj1xaj3pJTk62WrVqZb3%2B%2BuvWV199ZSUmJlqBgYHWkSNHLMuyrPj4eJdvzf3v//6v5ePjYy1evNg6cOCAtXjxYsvX19fKysry1FtoMO6Ozfz5863Nmzdb3333nZWTk2M99NBDlq%2Bvr7Vr1y5PvYUGc%2BbMGSsnJ8fKycmxJFnLli2zcnJyrB9%2B%2BMGyLMuaPXu2FR8f76z//vvvrauvvtqaPn269dVXX1mvv/661apVK%2Bt//ud/PPUWGpS74/PCCy9Yqamp1jfffGPt37/fmj17tiXJSklJ8dRbaDC/%2B93vLLvdbm3bts3Kz893Lv/3f//nrGmpx50rGZuWdNyZM2eOlZGRYR0%2BfNjau3evNXfuXOuqq66ytmzZYllWy/zcELCauJdeesnq3Lmz5efnZ/Xt29flK8EDBgywxo8f71L//vvvW927d7datWpl3Xzzzc3yj0AVd8YmMTHRuuGGGyw/Pz%2Brffv2VlxcnLVz504PdN3wqm4rcOlSNR7jx4%2B3BgwY4PKcbdu2WX369LH8/PysG2%2B80Vq5cmXjN95I3B2fJUuWWF26dLECAgKs4OBg67bbbrM%2B/PBDzzTfwGoaF0nW6tWrnTUt9bhzJWPTko47Dz/8sPN43L59e2vw4MHOcGVZLfNzY7Osf1zNCgAAACO4BgsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAAA8IiMjQyNHjpTD4ZDNZtP69evdev78%2BfNls9mqLYGBgQ3Ucf0RsAAAgEeUlpaqd%2B/eWrFixRU9f%2BbMmcrPz3dZIiIi9Otf/9pwp%2B4jYAEAAI8YNmyYnnnmGd177701bi8vL9esWbN0/fXXKzAwUP369dO2bduc26%2B55hqFhoY6lxMnTuirr77ShAkTGukd1M7X0w0AAADU5KGHHtKRI0eUnJwsh8Oh1NRUDR06VPv27VPXrl2r1a9atUrdunVT//79PdCtK2awAABAk/Pdd9/p3Xff1fvvv6/%2B/furS5cumjlzpm677TatXr26Wn1ZWZneeeedJjF7JTGDBQAAmqAvvvhClmWpW7duLuvLysrUtm3bavXr1q3TmTNn9OCDDzZWi5dFwAIAAE3Ozz//LB8fH2VnZ8vHx8dl2zXXXFOtftWqVRoxYoRCQ0Mbq8XLImABAIAmp0%2BfPqqsrFRhYWGd11QdPnxYW7du1YYNGxqpu7oRsAAAgEecPXtW3377rfPx4cOHlZubq%2Buuu07dunXTb37zGz344IN6/vnn1adPHxUVFenTTz9Vr169NHz4cOfz3njjDYWFhWnYsGGeeBs1slmWZXm6CQAA0PJs27ZNAwcOrLZ%2B/PjxWrNmjSoqKvTMM8/orbfe0t/%2B9je1bdtWMTExWrBggXr16iXpwqnEzp0768EHH9Szzz7b2G%2BhVgQsAAAAw/4fK5gJ6GWDdtoAAAAASUVORK5CYII%3D"/>
        </div>
        <div role="tabpanel" class="tab-pane col-md-12" id="common2359107911530742763">
            
<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">528860</td>
        <td class="number">739</td>
        <td class="number">0.1%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">15440</td>
        <td class="number">683</td>
        <td class="number">0.1%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1990543</td>
        <td class="number">636</td>
        <td class="number">0.1%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1352131</td>
        <td class="number">583</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1506773</td>
        <td class="number">580</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">107970</td>
        <td class="number">580</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">2514814</td>
        <td class="number">555</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">593991</td>
        <td class="number">550</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1525686</td>
        <td class="number">548</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1295315</td>
        <td class="number">545</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="other">
        <td class="fillremaining">Other values (34317)</td>
        <td class="number">1220675</td>
        <td class="number">99.5%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr>
</table>
        </div>
        <div role="tabpanel" class="tab-pane col-md-12"  id="extreme2359107911530742763">
            <p class="h4">Minimum 5 values</p>
            
<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">109</td>
        <td class="number">2</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:9%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">344</td>
        <td class="number">4</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:18%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">2708</td>
        <td class="number">13</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:56%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">2732</td>
        <td class="number">18</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:78%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">3021</td>
        <td class="number">23</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr>
</table>
            <p class="h4">Maximum 5 values</p>
            
<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">30511912</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">30512279</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">30529526</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">30532527</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">30539573</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr>
</table>
        </div>
    </div>
</div>
</div>
    <div class="row headerrow highlight">
        <h1>Correlations</h1>
    </div>
    <div class="row variablerow">
    <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAmYAAAIaCAYAAACKzIEwAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4xLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvDW2N/gAAIABJREFUeJzt3XlclmWi//EvJLKpYLG4hKMhqGkJiGKlntxejuY2o8kc9ylwlMQlraajZcVYWjgzKVkec/KXNoW5jU5OdkZLnVJwQUdFDDQNFQUlUzbZnt8fHp/jE5jofeNzq5/368UruO6Fi%2BsP%2Bfa97ufBxWaz2QQAAACnc3X2BAAAAHAZwQwAAMAiCGYAAAAWQTADAACwCIIZAACARRDMAAAALIJgBgAAYBEEMwAAAIsgmAEAAFhEHWdPAID5Tpw4oZ49e17zuJubm%2BrVq6fmzZvr8ccf18iRI1WvXr1bOEMAQHVc%2BJNMwJ3n6mAWGhpaJXSVlZUpPz9fJ0%2BelCQ1adJES5cu1S9%2B8YtbPlcAwP8hmAF3oKuD2YcffqioqKhqz0tJSVFcXJwKCgoUHh6uTz755FZOEwDwEzxjBtzFoqKi9Oyzz0qS0tLSdODAASfPCADubgQz4C7Xu3dv%2B%2Bf79u1z4kwAADz8D9zl6tevb/%2B8sLDQ4djOnTu1bNky7dmzR%2BfPn1eDBg0UFhamUaNG6ZFHHqn2fhcuXNAnn3yiLVu2KCsrSwUFBfL09FSzZs3UvXt3jR49Wj4%2BPg7XtGrVSpL09ddfa86cOdq0aZNcXV3Vtm1b/eUvf1GdOnW0b98%2BLV26VOnp6crJyZG7u7tatGihXr16afjw4dW%2BeKGkpESffPKJNmzYoKysLJWVlSkwMFCPPvqonnrqKTVv3tzh/JSUFI0ePVrt27fXRx99pGXLlmnt2rU6fvy43Nzc1LZtW40aNUq9evW6maUGgOsimAF3uePHj9s/b9Sokf3zxMRELV68WJLk4%2BOj0NBQ5ebmatOmTdq0aZNiYmL03HPPOdzr2LFjGjt2rHJyclSnTh01a9ZMTZs21cmTJ3Xw4EEdPHhQn332mVatWiVvb%2B8qc4mPj1daWppCQ0OVn58vf39/1alTR1988YWmTp2q8vJyNWzYUC1btlRhYaH%2B/e9/a9%2B%2BfVq3bp0%2B%2BeQTh3B2%2BvRp/fa3v9XRo0clSc2bN5e3t7eOHDmi5ORkrV27VnPmzFG/fv2qzKOsrEyxsbHavn27GjZsqODgYH333XfasWOHduzYoVdeeUX/%2BZ//aWzhAaA6NgB3nOzsbFtoaKgtNDTUtmPHjp899/nnn7eFhoba2rZta8vLy7PZbDbbxx9/bAsNDbVFRkba/va3v9nPraystH322We2sLAwW2hoqG3FihUO9xo5cqQtNDTUNmzYMNuZM2ccrluzZo2tdevWttDQUNvy5csdrrsy13bt2tlSU1NtNpvNVlFRYfvhhx9sFRUVtscee8wWGhpqW7x4sa28vNx%2B3YEDB2ydO3e2hYaG2hYtWmQfLy8vtw0aNMgWGhpq69Onj%2B3QoUP2YxcvXrTNmDHD/jPv3bvXfmzHjh32uYSFhdnWrVtnP3bhwgXbmDFjbKGhobZOnTrZysrKfnZdAeBm8IwZcBcqKSlRenq6Zs2apbVr10qSxo4dKz8/P5WWlmrBggWSpNdff10DBw60X%2Bfi4qJ%2B/frZm7IFCxaovLxcknTu3DllZmZKkhISEhQQEOBw3eDBg9WpUydJ0uHDh6udV9%2B%2BfdWxY0dJkqurq3x9fZWfn6%2B8vDxJ0rBhw3TPPffYz2/btq2mTp2qXr16ydfX1z7%2B%2Beef69ChQ3J3d9fixYvVunVr%2B7F69erpD3/4g7p27aqysjL96U9/qnYukyZN0oABA%2Bxf169f3/5znz9/Xt999901VhcAbh5bmcAdbvTo0dc958knn9TkyZMlXX515tmzZ%2BXt7X3NN6kdOHCgEhISdObMGaWnp%2Bvhhx/Wfffdpx07dqikpEQeHh5VrqmoqLBvNZaUlFR73w4dOlQZa9iwoXx8fPTjjz9q%2BvTpmjBhgtq3by9X18v/Xzls2DANGzbM4ZrNmzdLknr06KGgoKBqv9dvf/tbbdu2Tampqbp48aLDs3aS1L179yrXBAcH2z%2B/cOFCtfcFACMIZsAd7qdvMOvi4iJ3d3f5%2BvqqVatW6tWrl1q2bGk/fqX1Kisr04gRI65533vuuUeVlZU6evSoHn74Yfu4h4eHcnJytG/fPn3//ffKzs7WkSNHdOjQIRUVFUmSKisrq72nv79/td9n%2BvTpeumll7RlyxZt2bJFPj4%2BioqK0mOPPabHH3/c4dk4SfY2q23bttec/5VjFRUVOn78uNq1a%2BdwPDAwsMo1VwfOioqKa94bAG4WwQy4w82cOfOabzBbnYsXL0qSSktLtWfPnuuef3VzdPToUb355pvasmWLQ/iqV6%2BeIiMjlZubq4yMjGveq7qmTbrciv3iF7/QBx98oG%2B%2B%2BUY//vijvvjiC33xxRdycXHR448/rldeecUe0AoKCiSpSgt2tavD6k9fjSpd/rNVP8fGe3MDqAUEMwAOPD09JV1ulFavXl3j686dO6eRI0fq3LlzatKkiYYNG6YHH3xQDzzwgO6//365uLho2rRpPxvMfk5UVJSioqJUUlKiXbt2aefOndq2bZsOHjyoL7/8Ujk5OVq7dq1cXFzsr/i8EjKrc3WgrO4VogDgDAQzAA5atGgh6fJbX5SXl6tOnar/TNhsNqWkpKhRo0Zq0qSJ6tatq1WrVuncuXPy9fXVqlWrdO%2B991a57syZMzc8n9LSUmVnZ6ugoEDt27eXh4eHunTpoi5dumjq1Kn67LPP9OyzzyojI0OHDx9W69at9cADDyg9PV0HDx685n33798v6fLWbrNmzW54XgBQG3hVJgAHHTt2VP369VVYWHjNxmz9%2BvUaM2aM%2Bvbtq9OnT0u6/Pc5pct/EL26UJaVlaW9e/dKurHns7Zu3ap%2B/fpp3LhxKi0trXL80UcftX9%2B5b5XHtzfvHmzsrOzq73vhx9%2BKEkKCwtTgwYNajwfAKhNBDMADry8vDRu3DhJ0uzZs7Vq1SqH58X%2B%2Bc9/atasWZIuv73FlbbpgQcekCRlZGRo48aN9vNtNpu2bt2qmJgYlZWVSZKKi4trPJ9u3bqpYcOGOn/%2BvF544QWdP3/efqywsFBz586VJDVu3FghISGSpF/%2B8pdq1aqVLl26pNjYWIft04KCAr300kv617/%2BpTp16mj69Ok1XxwAqGVsZQKoIjY2VtnZ2VqxYoX%2B67/%2BS2%2B99Zbuv/9%2BnTlzRrm5uZKkiIgI/eEPf7BfM3ToUP31r3/V8ePHNWnSJDVt2lQNGzZUTk6Ozp07Jzc3N3Xq1Empqak3tKVZt25dvf3223r66ae1YcMGbdq0Sc2aNZOrq6uys7NVVFQkT09PzZkzR3Xr1pUk1alTRwsXLlRsbKyOHj2qQYMGObzz/5W39Hj11VcVGRlp7uIBgAEEMwBVuLi4KCEhQX369NEnn3yivXv32t%2BwNSwsTP3791d0dLQ9CEmXX%2BW4cuVKLV68WF9%2B%2BaVOnDihs2fPqlGjRnr88cc1ZswYeXl5qVevXsrIyNCpU6fUpEmTGs0nKipKn376qT744APt3r1bx44dU506ddSoUSN16dJFTz31VJV73X///Vq1apU%2B/vhjff755zpy5IhOnz6txo0bq2vXrhoxYkSVv5UJAM7mYuM13wAAAJbAM2YAAAAWQTADAACwCIIZAACARRDMAADAXSE/P1%2B9e/dWSkrKNc/ZsmWLBgwYoLCwMPXt21dffvmlw/HFixerW7duCgsL06hRo3T06FFT50gwAwAAd7zdu3crOjpa33///TXPOXbsmOLj4zV58mTt2rVL8fHxmjJliv0tftasWaNly5ZpyZIlSklJUdu2bTVp0iRT/3YuwQwAANzR1qxZo%2BnTp2vq1KnXPS8yMlK9evVSnTp11K9fP3Xs2FHJycmSpBUrVmj48OEKCQmRu7u7pk2bplOnTv1sA3ejCGYAAMDScnNzdfDgQYePK292XRNdunTR//zP/6hfv34/e15WVpZCQ0Mdxlq2bGn/6yE/Pe7m5qbmzZs7/HURo3iDWSNcXJw9A%2BDO0KKFlJkphYRI333n7NkAtz9nvUVpLf1eTJ4/X0lJSQ5jEydOVHx8fI2u9/f3r9F5hYWF8vT0dBjz8PBQUVFRjY6bgWAGwPl8faV77rn8XwD4iejoaPXo0cNhrKZh60Z4enqqpKTEYaykpETe3t41Om4GghkAADCHa%2B08IRUQEKCAgIBauffVQkNDdfDgQYexrKwstWvXTpIUEhKizMxMde/eXZJUVlamY8eOVdn%2BNIJnzAAAgDlcXWvn4xYZOHCgUlNTtWHDBpWXl2vDhg1KTU3VoEGDJElDhgzR8uXLlZGRoUuXLmnevHny8/NTZGSkaXMgmAEAgLtWeHi41q1bJ0kKDg7WO%2B%2B8o0WLFqljx45auHChFixYoBYtWkiShg4dqrFjx%2BqZZ55R586dlZ6erkWLFsnNzc20%2BfBHzI3g4X/AHOHh0p49UkSElJbm7NkAtz9n/Wp3d6%2Bd%2B166VDv3tSAaMwAAAIvg4X8AAGCOW/g82J2KYAYAAMxBMDOMFQQAALAIGjMAAGAOGjPDWEEAAACLoDEDAADmoDEzjGAGAADMQTAzjBUEAACwCBozAABgDhozw1hBAAAAi6AxAwAA5qAxM4xgBgAAzEEwM4wVBAAAsAgaMwAAYA4aM8NYQQAAAIugMQMAAOagMTOMYAYAAMxBMDOMFQQAALAIGjMAAGAOGjPDWEEAAACLoDEDAADmoDEzjGAGAADMQTAzjBUEAACwCBozAABgDhozw1hBAAAAi6AxAwAA5qAxM4xgBgAAzEEwM4wVBAAAsAgaMwAAYA4aM8NYQQAAAIugMQMAAOagMTOMYAYAAMxBMDOMFQQAALAIGjMAAGAOGjPDWEEAAACLoDEDAADmoDEzjGAGAADMQTAzjBUEAACwCBozAABgDhozwwhmAADAHAQzw1hBAAAAi6AxAwAA5qAxM4xgBgAA7mjnzp3TSy%2B9pNTUVN1zzz0aOHCgXnjhBdWp4xiDYmJitHv3boexoqIiRUdH67XXXtPZs2f12GOPycvLy368YcOG2rx5s2lzJZgBAABzWLQxmzJligIDA7Vt2zadPXtWEyZM0NKlSxUTE%2BNw3vvvv%2B/w9cqVK5WUlKSJEydKkvbv36%2BmTZuaGsR%2ByporCAAAbj%2BurrXzYcDx48eVmpqq5557Tp6engoKClJcXJw%2B%2Buijn73u6NGjSkhIUGJiogICAiRdDmbt2rUzNJ/roTEDAACWlpubq7y8PIcxf39/e2D6OZmZmfL19VVgYKB9LDg4WKdOndKFCxfUoEGDaq979dVXNXjwYEVGRtrH9u/frx9//FH9%2B/fX2bNn9dBDD%2BmFF15Qy5Ytb/Inq4pgBgAAzFFLW5nJyclKSkpyGJs4caLi4%2BOve21hYaE8PT0dxq58XVRUVG0w27Vrl/bt26fExESH8QYNGqhly5aKjY1V3bp19fbbb%2Bu3v/2tNmzYoPr169/oj1UtghkAALC06Oho9ejRw2HM39%2B/Rtd6eXmpuLjYYezK197e3tVek5ycrL59%2B1b5HvPmzXP4%2BsUXX9SqVau0a9cude/evUbzuR6CGQAAMEctNWYBAQE12rasTkhIiM6fP6%2BzZ8/Kz89PknTkyBE1atSo2parvLxcmzZt0jvvvOMwXlBQoHfeeUcjR45U06ZNJUkVFRUqLy%2BXh4fHTc2tOjz8DwAAzGHBh/%2BbN2%2BuDh066PXXX1dBQYGys7O1cOFCDR06tNrzDx8%2BrEuXLikiIsJhvF69evrmm280d%2B5cXbx4UYWFhUpISND999/v8ByaUQQzAABwR5s/f77Ky8vVs2dPDRs2TF27dlVcXJwkKTw8XOvWrbOfm52dLR8fH7m7u1e5z8KFC1VZWalevXqpa9euysvL0%2BLFi%2BXm5mbaXF1sNpvNtLvdbVxcnD0D4M4QHi7t2SNFREhpac6eDXD7c9av9j59aue%2BGzfWzn0tiMYMAADAInj4HwAAmMOi7/x/OyGYAQAAcxDMDGMFAQAALILGDAAAmIPGzDBWEAAAwCJozAAAgDlozAwjmAEAAHMQzAxjBQEAACyCxgwAAJiDxswwVhAAAMAiaMwAAIA5aMwMI5gBAABzEMwMYwUBAAAsgsYMAACYg8bMMFYQAADAImjMAACAOWjMDCOYAQAAcxDMDGMFAQAALILGDAAAmIPGzDBWEAAAwCJozAAAgDlozAwjmAEAAHMQzAxjBQEAACyCxgwAAJiDxswwVhAAAMAiaMwAAIA5aMwMI5gBAABzEMwMYwUBAAAsgsYMAACYg8bMMIIZAAAwB8HMMFYQAADAImjMAACAOWjMDGMFAQAALILGDAAAmIPGzDCCGQAAMAfBzDBWEAAAwCJozAAAgDlozAxjBQEAACyCxgwAAJiDxswwghkAADAHwcwwVhAAAMAiaMwAAIA5aMwMYwUBAMAd7dy5c4qLi1NkZKSioqI0e/ZslZeXV3tuTEyMHnroIYWHh9s/tm7dKkmqqKjQ3Llz9eijjyo8PFwTJkxQbm6uqXMlmAEAAHO4utbOh0FTpkyRl5eXtm3bppUrV2r79u1aunRpteceOHBAS5YsUVpamv2jW7dukqR3331XX3/9tVatWqVt27bJw8NDM2fONDy/qxHMAACAOSwYzI4fP67U1FQ999xz8vT0VFBQkOLi4vTRRx9VOTc7O1s//vijHnzwwWrv9emnnyo2NlaNGzdWvXr1NGPGDG3dulXZ2dmG5ng1njEDAACWlpubq7y8PIcxf39/BQQEXPfazMxM%2Bfr6KjAw0D4WHBysU6dO6cKFC2rQoIF9fP/%2B/fL29tbUqVO1f/9%2B%2Bfn5aezYsRo6dKguXryo06dPKzQ01H6%2Bn5%2BffHx8dPjwYQUFBZnwkxLMAACAWWrp4f/k5GQlJSU5jE2cOFHx8fHXvbawsFCenp4OY1e%2BLioqcghmpaWlCgsL09SpUxUSEqKUlBTFx8fL29tb4eHhkiQvLy%2BHe3l4eKiwsPCmfq7qEMwAAIClRUdHq0ePHg5j/v7%2BNbrWy8tLxcXFDmNXvvb29nYYHzx4sAYPHmz/ukuXLho8eLD%2B8Y9/6NFHH3W49oqSkpIq9zGCYAYAAMxRS41ZQEBAjbYtqxMSEqLz58/r7Nmz8vPzkyQdOXJEjRo1Uv369R3OXblypby9vdW3b1/7WGlpqdzd3eXj46PAwEBlZWXZtzPz8vJ0/vx5h%2B1No3j4HwAAmMOCD/83b95cHTp00Ouvv66CggJlZ2dr4cKFGjp0aJVzCwoKlJCQoPT0dFVWVuqrr77S3//%2Bd0VHR0uSfv3rX%2Bvdd99Vdna2CgoK9Prrr6tTp05q1qyZoTlejcYMAADc0ebPn6/XXntNPXv2lKurqwYPHqy4uDhJUnh4uF599VUNHDhQY8aMUVFRkSZOnKhz584pKChIc%2BfOVWRkpCTpmWeeUXl5uUaMGKHCwkJFRUXpz3/%2Bs6lzdbHZbDZT73g3cXFx9gyAO0N4uLRnjxQRIaWlOXs2wO3PWb/a33ijdu774ou1c18LYisTAADAItjKBAAA5uBvZRpGMAMAAOYgmBnGCgIAAFgEjRkAADAHjZlhrCAAAIBF0JgBAABz0JgZRjADAADmIJgZxgoCAABYBI0ZAAAwB42ZYawgAACARdCYAQAAc9CYGUYwAwAA5iCYGcYKAgAAWASNGQAAMAeNmWGsIAAAgEXQmAEAAHPQmBlGMAMAAOYgmBnGCgIAAFgEjRkAADAHjZlhrCAAAIBF0JgBAABz0JgZRjADAADmIJgZxgoCAABYBI0ZAAAwB42ZYQQzAABgDoKZYawgAACARdCYAQAAc9CYGcYKAgAAWASNGQAAMAeNmWEEMwAAYA6CmWGsIAAAgEXQmAEAAHPQmBnGCgIAAFgEjRkAADAHjZlhBDMAAGAOgplhrCAAAIBF0JgBAABz0JgZxgoCAABYBI0ZAAAwB42ZYQQzAABgDoKZYawgAACARdCYAQAAc9CYGUYwAwAAd7Rz587ppZdeUmpqqu655x4NHDhQL7zwgurUqRqDPv74Yy1dulS5ubkKCAjQ6NGjNWLECElSZWWlOnToIJvNJhcXF/s1X3/9tby8vEyZK8EMAACYw6KN2ZQpUxQYGKht27bp7NmzmjBhgpYuXaqYmBiH8/75z3/qj3/8oxYvXqz27dtr7969GjdunPz8/NSnTx9lZWWprKxMe/bsUd26dWtlrgQzAABgjloKZrm5ucrLy3MY8/f3V0BAwHWvPX78uFJTU7V161Z5enoqKChIcXFxeuutt6oEszNnzig2NlZhYWGSpPDwcEVFRWnnzp3q06eP9u/fr1atWtVaKJMIZgAAwOKSk5OVlJTkMDZx4kTFx8df99rMzEz5%2BvoqMDDQPhYcHKxTp07pwoULatCggX38ypblFefOndPOnTv14osvSpL279%2BvS5cuaciQITp58qSCg4M1bdo0RUREGPnxHBDMAACAOWqpMYuOjlaPHj0cxvz9/Wt0bWFhoTw9PR3GrnxdVFTkEMyulpeXp9/97ndq166d%2BvfvL0ny8PDQww8/rMmTJ8vHx0cfffSRnn76aa1bt05BQUE3%2BmNVi2AGAAAsLSAgoEbbltXx8vJScXGxw9iVr729vau9Zu/evZo8ebIiIyP1xhtv2F8k8Pvf/97hvKefflqrV6/Wli1bNHLkyJua309Z8yk9AABw%2B3F1rZ0PA0JCQnT%2B/HmdPXvWPnbkyBE1atRI9evXr3L%2BypUrNXbsWI0ZM0bz5s1zeJ7sT3/6k9LT0x3OLy0tlbu7u6E5Xo1gBgAAzGHBYNa8eXN16NBBr7/%2BugoKCpSdna2FCxdq6NChVc7duHGjXnnlFS1YsEBPPfVUlePffvutZs%2Berby8PJWWliopKUkFBQXq3bu3oTlejWAGAADuaPPnz1d5ebl69uypYcOGqWvXroqLi5N0%2BZWX69atkyQlJSWpoqJCkyZNUnh4uP3j5ZdfliS98cYbatasmQYNGqSoqCilpqbqgw8%2BkK%2Bvr2lzdbHZbDbT7na3uerN5QAYEB4u7dkjRURIaWnOng1w%2B3PWr/bt22vnvo88Ujv3tSAaMwAAAIvgVZkAAMAcFn3n/9sJwQwAAJiDYGYYKwgAAGARNGYAAMAcNGaGsYIAAAAWQWMGAADMQWNmGMEMAACYg2BmGCsIAABgETRmAADAHDRmhrGCAAAAFkFjBgAAzEFjZhjBDAAAmINgZhgrCAAAYBE0ZgAAwBw0ZoYRzAAAgDkIZoaxggAAABZBYwYAAMxBY2YYKwgAAGARNGYAAMAcNGaGEcwAAIA5CGaGsYIAAAAWQWMGAADMQWNmGCsIAABgETRmAADAHDRmhhHMAACAOQhmhrGCAAAAFkFjBgAAzEFjZhgrCAAAYBE0ZgAAwBw0ZoYRzAAAgDkIZoaxggAAABZBYwYAAMxBY2YYKwgAAGARNGYAAMAcNGaGEcwAAIA5CGaGsYIAAAAWQWMGAADMQWNmGCsIAABgETRmAADAHDRmhhHMAACAOQhmhrGCAAAAFlHrjVlFRYVOnTqloKCg2v5WAADAmWjMDLuhFTxx4oRatWqlEydOKDw8XLt27bruNVOnTtXatWslSadOnVJ4eLhOnTp1c7O9ATExMXrvvfeuebxVq1ZKSUmp9XkAAADnOnfunOLi4hQZGamoqCjNnj1b5eXl1Z67ZcsWDRgwQGFhYerbt6%2B%2B/PJLh%2BOLFy9Wt27dFBYWplGjRuno0aOmzvWmo21aWpoiIyOve94PP/xg/7xJkyZKS0tTkyZNbvbb1tj777%2Bv8ePH1/r3AQAA/8vVtXY%2BDJoyZYq8vLy0bds2rVy5Utu3b9fSpUurnHfs2DHFx8dr8uTJ2rVrl%2BLj4zVlyhSdOXNGkrRmzRotW7ZMS5YsUUpKitq2batJkybJZrMZnuMVN72V2apVK3344YeKiorSxo0bNX/%2BfJ0%2BfVoBAQEaMGCA4uLiNGPGDO3atUtpaWk6ePCgZs6cqZ49e2rTpk26//771apVK82cOVPLly9Xbm6uWrVqpVdffVWtWrWSJH3zzTd688039f333ys0NFQdOnTQv//9by1btuy68xs1apQ6deqk%2BPh4lZWVKTExUWvXrpWLi4tiYmJu%2BOfNzc1VXl6ew5h/ixYK8PW94XsB%2BInWrR3/C%2BDmpaU573vX0lZmtb%2BD/f0VEBBw3WuPHz%2Bu1NRUbd26VZ6engoKClJcXJzeeuutKnlgzZo1ioyMVK9evSRJ/fr10%2BrVq5WcnKxJkyZpxYoVGj58uEJCQiRJ06ZN04oVK5SSkqLOnTub8rMafsaspKREzz33nBYvXqyoqCilp6drxIgR6tKli2bPnq3vv//eHpBOnDhR5frPPvtMy5cvl4eHhyZNmqQ333xTS5Ys0YkTJzR%2B/HjNmDFDQ4YM0d69ezV%2B/Hi1adPmhue4cOFCffXVV1q5cqXuu%2B8%2BvfLKKzd8j%2BTkZCUlJTmMTZw8WfGTJ9/wvQBcw1//6uwZALc/Fxdnz8B01f4OnjhR8fHx1702MzNTvr6%2BCgwMtI8FBwfr1KlTunDhgho0aGAfz8rKUmhoqMP1LVu2VEZGhv14bGys/Zibm5uaN2%2BujIwM6wQzSfLw8NDKlStVWVmpiIgI7d69W641TM2jRo2Sv7%2B/JKlv375atGiRJGn9%2BvVq06aNoqOjJUmRkZEaNmyY9u/ff8Pz%2B9vf/qbx48fbX4Awc%2BZMrVu37obuER0drR49ejiM%2BQ8YIP2//3fD8wHwE61bXw5lw4dL//sPIIDbj021Ewqr/R38v9nhegoLC%2BXp6ekwduXroqIih2BW3bkeHh4qKiqq0XEzGA5mHh4e%2Bvjjj7Vw4UJNmzZNBQUF6tOnj2bOnCkfH5/rXu/n5/d/k6lTx75Pm5OTo6ZNmzqcGxQUdFPBLDc3V40bN7Z/3aBBgxrN7WoBAQFVK9PvvrvhuQD4GRkZzt2GAWBJ1f4OriEvLy8VFxc7jF352ttRXzq8AAAauklEQVTb22Hc09NTJSUlDmMlJSX286533AyGN4MLCgqUm5urefPm6ZtvvlFycrIOHDjws6%2BIrImmTZtWefXmzb6as1GjRsrOzrZ/XVRUpIsXLxqaHwAAcFRZWTsfRoSEhOj8%2BfM6e/asfezIkSNq1KiR6tev73BuaGioMjMzHcaysrLsz5SFhIQ4HC8rK9OxY8eqbH8aYTiYFRYWKjY2VuvXr5fNZlNAQIBcXV3VsGFDSVLdunVvKgQNGjRIhw4d0tq1a1VRUaF9%2B/ZpxYoVNzXHJ598Uu%2B//76OHDmiS5cuac6cOaqoqLipewEAgOpZMZg1b95cHTp00Ouvv66CggJlZ2dr4cKFGjp0aJVzBw4cqNTUVG3YsEHl5eXasGGDUlNTNWjQIEnSkCFDtHz5cmVkZOjSpUuaN2%2Be/Pz8avQuFTVlOJgFBgZq/vz5Wrx4sSIiItS/f3917txZY8eOlSQNHjxYq1at0vDhw2/ovo0aNbLfNzIyUnPnzlWXLl3k5uZ2w3OMjY3VwIEDNXLkSHXp0kX169eXL6%2BmBADgrjB//nyVl5erZ8%2BeGjZsmLp27aq4uDhJUnh4uP258%2BDgYL3zzjtatGiROnbsqIULF2rBggVq0aKFJGno0KEaO3asnnnmGXXu3Fnp6elatGjRTWWTa3GxmfnmGybKycnRDz/8oAcffNA%2BNmfOHOXl5WnevHlOnNlV7sBXvgBOER4u7dkjRUTwjBlgBif9ar90qXbu6%2B5eO/e1Isv%2B7YQffvhBw4cP14EDByRJGRkZWrdunbp37%2B7kmQEAANSOWv9bmTfrwQcf1IwZM/Tss88qLy9Pfn5%2BGjdunPr376/Zs2dr5cqV17z2d7/7He/6DwDALWb0eTBYeCvztsBWJmAOtjIBcznpV3thYe3c18R3o7A8y25lAgAA3G0su5UJAABuL2xlGkcwAwAApiCYGcdWJgAAgEXQmAEAAFPQmBlHYwYAAGARNGYAAMAUNGbGEcwAAIApCGbGsZUJAABgETRmAADAFDRmxtGYAQAAWASNGQAAMAWNmXEEMwAAYAqCmXFsZQIAAFgEjRkAADAFjZlxNGYAAAAWQWMGAABMQWNmHMEMAACYgmBmHFuZAAAAFkFjBgAATEFjZhyNGQAAgEXQmAEAAFPQmBlHMAMAAKYgmBnHViYAAIBF0JgBAABT0JgZR2MGAABgETRmAADAFDRmxhHMAACAKQhmxrGVCQAAYBE0ZgAAwBQ0ZsbRmAEAAFgEjRkAADAFjZlxBDMAAGAKgplxbGUCAABYBI0ZAAAwBY2ZcTRmAAAAFkFjBgAATEFjZhzBDAAAmIJgZhxbmQAAABZBYwYAAExBY2YcwQwAANy1ioqKlJCQoM2bN6u8vFw9e/bUrFmz5O3tXe35Gzdu1MKFC5WdnS1fX1/9%2Bte/VlxcnFxdL29C9u3bV6dOnbJ/LUkrV65UcHBwjeZDMAMAAKa4HRuzhIQE5eTkaOPGjaqoqNCUKVOUmJioWbNmVTn3wIEDev755/XnP/9Z//Ef/6HvvvtOsbGx8vLy0lNPPaWCggJ999132rRpk5o2bXpT8%2BEZMwAAYIrKytr5yM3N1cGDBx0%2BcnNzDc%2B3uLhY69ev16RJk%2BTr66v77rtP06dP1%2BrVq1VcXFzl/JMnT%2Bo3v/mNunfvLldXVwUHB6t3797auXOnpMvBzdfX96ZDmURjBgAALC45OVlJSUkOYxMnTlR8fPx1ry0pKdGZM2eqPVZcXKyysjKFhobax4KDg1VSUqJjx46pTZs2Duf36dNHffr0cbj3V199pQEDBkiS9u/fL09PT40cOVKZmZlq2rSp4uPj1b179xr/rAQzAABgitrayoyOjlaPHj0cxvz9/Wt07b59%2BzR69Ohqj02ePFmS5OXlZR/z9PSUJBUWFv7sfQsKCjR58mR5eHho7NixkiQXFxc99NBDevbZZ9WkSRN9/vnnio%2BP1/LlyxUWFlaj%2BRLMAACApQUEBCggIOCmro2KitLhw4erPZaenq63335bxcXF9of9r2xh1qtX75r3PHr0qCZNmqT77rtPH374of3cmJgYh/MGDhyov//979q4cWONgxnPmAEAAFPU1jNmtaVFixZyc3NTVlaWfezIkSNyc3NT8%2BbNq71my5YtevLJJ9W1a1ctWbJEPj4%2B9mNLlizR9u3bHc4vLS2Vu7t7jedEMAMAAKa43YKZp6en%2Bvbtq8TEROXn5ys/P1%2BJiYnq37%2B/PDw8qpy/d%2B9ePfPMM3rxxRf1wgsvqE4dx43HnJwcvfrqq8rOzlZ5eblWrlyptLQ0/epXv6rxnFxsNpvN8E92t3JxcfYMgDtDeLi0Z48UESGlpTl7NsDtz0m/2v/nf2rnvr171859pcvPis2dO1ebN29WWVmZevbsqZdeesn%2B3NkTTzyhAQMGaPz48Ro/fry%2B%2Buor%2B3NoV3To0EHvv/%2B%2BSktLlZiYqH/84x%2B6ePGiWrZsqeeee05RUVE1ng/BzAiCGWAOghlgLif9at%2B4sXbue9ULIe94PPwPAABMcTu%2BwazV8IwZAACARdCYAQAAU9CYGUdjBgAAYBE0ZgAAwBQ0ZsYRzAAAgCkIZsaxlQkAAGARNGYAAMAUNGbG0ZgBAABYBI0ZAAAwBY2ZcQQzAABgCoKZcWxlAgAAWASNGQAAMAWNmXE0ZgAAABZBYwYAAExBY2YcwQwAAJiCYGYcW5kAAAAWQWMGAABMQWNmHI0ZAACARdCYAQAAU9CYGUcwAwAApiCYGcdWJgAAgEXQmAEAAFPQmBlHYwYAAGARNGYAAMAUNGbGEcwAAIApCGbGsZUJAABgETRmAADAFDRmxtGYAQAAWASNGQAAMAWNmXEEMwAAYAqCmXFsZQIAAFgEjRkAADAFjZlxNGYAAAAWQWMGAABMQWNmHMEMAACYgmBmHFuZAAAAFkFjBgAATEFjZhyNGQAAgEXQmAEAAFPQmBlHMAMAAKYgmBnHViYAAIBFEMwAAIApKitr56M2FRUV6cUXX1RUVJQ6dOig559/XoWFhdc8f9asWWrXrp3Cw8PtH8nJyfbjixcvVrdu3RQWFqZRo0bp6NGjNzQfghkAADDF7RjMEhISlJOTo40bN%2BqLL75QTk6OEhMTr3n%2B/v37lZCQoLS0NPtHdHS0JGnNmjVatmyZlixZopSUFLVt21aTJk2SzWar8XwIZgAA4K5UXFys9evXa9KkSfL19dV9992n6dOna/Xq1SouLq5yfmlpqb799lu1a9eu2vutWLFCw4cPV0hIiNzd3TVt2jSdOnVKKSkpNZ4TD/8DAABT1Fa7lZubq7y8PIcxf39/BQQEXPfakpISnTlzptpjxcXFKisrU2hoqH0sODhYJSUlOnbsmNq0aeNwfkZGhsrLyzV//nzt3r1b9evX15AhQxQTEyNXV1dlZWUpNjbWfr6bm5uaN2%2BujIwMde7cuUY/K8EMAABYWnJyspKSkhzGJk6cqPj4%2BOteu2/fPo0ePbraY5MnT5YkeXl52cc8PT0lqdrnzC5evKhOnTpp1KhR%2BuMf/6hDhw7pmWeekaurq2JiYlRYWGi//goPDw8VFRVdd55XEMwAAIApaqsxi46OVo8ePRzG/P39a3RtVFSUDh8%2BXO2x9PR0vf322youLpa3t7ck2bcw69WrV%2BX8xx57TI899pj964cfflhjxozRhg0bFBMTI09PT5WUlDhcU1JSYr93TRDMAACAKWormAUEBNRo2/JGtWjRQm5ubsrKylL79u0lSUeOHLFvQf7UP//5T509e1a/%2Bc1v7GOlpaXy8PCQJIWEhCgzM1Pdu3eXJJWVlenYsWMOW6XXw8P/AADgruTp6am%2BffsqMTFR%2Bfn5ys/PV2Jiovr3728PW1ez2Wx64403tH37dtlsNqWlpenDDz%2B0vypzyJAhWr58uTIyMnTp0iXNmzdPfn5%2BioyMrPGcaMwAAIApbsd3/p81a5bmzp2rAQMGqKysTD179tRLL71kP/7EE09owIABGj9%2BvHr37q0XX3xRr7zyis6cOSM/Pz/Fx8dr0KBBkqShQ4fq4sWLeuaZZ5Sfn6%2BHHnpIixYtkpubW43n42K7kTfXgCMXF2fPALgzhIdLe/ZIERFSWpqzZwPc/pz0q33GjNq57%2BzZtXNfK6IxAwAAprgdGzOrIZgBAABTEMyM4%2BF/AAAAi6AxAwAApqAxM47GDAAAwCJozAAAgClozIwjmAEAAFMQzIxjKxMAAMAiaMwAAIApaMyMozEDAACwCBozAABgChoz4whmAADAFAQz49jKBAAAsAgaMwAAYAoaM%2BNozAAAACyCxgwAAJiCxsw4ghkAADAFwcw4tjIBAAAsgsYMAACYgsbMOBozAAAAi6AxAwAApqAxM45gBgAATEEwM46tTAAAAIugMQMAAKagMTOOxgwAAMAiaMwAAIApaMyMI5gBAABTEMyMYysTAADAImjMAACAKWjMjKMxAwAAsAgaMwAAYAoaM%2BMIZgAAwBQEM%2BPYygQAALAIGjMAAGAKGjPjCGYAAMAUBDPj2MoEAACwCBozAABgChoz42jMAAAALILGDAAAmILGzDiCGQAAMAXBzDi2MgEAACyCxgwAAJiCxsw4GjMAAACLoDEDAACmuB0bs6KiIiUkJGjz5s0qLy9Xz549NWvWLHl7e1c59%2BWXX9b69esdxkpKSvToo49qyZIlqqysVIcOHWSz2eTi4mI/5%2Buvv5aXl1eN5kMwAwAAprgdg1lCQoJycnK0ceNGVVRUaMqUKUpMTNSsWbOqnPvaa6/ptddes3/9r3/9S9OmTdPvf/97SVJWVpbKysq0Z88e1a1b96bmw1YmAAC4KxUXF2v9%2BvWaNGmSfH19dd9992n69OlavXq1iouLf/ba/Px8TZ8%2BXTNmzFBISIgkaf/%2B/WrVqtVNhzKJxgwAAJikthqz3Nxc5eXlOYz5%2B/srICDguteWlJTozJkz1R4rLi5WWVmZQkND7WPBwcEqKSnRsWPH1KZNm2veNzExUe3atdPAgQPtY/v379elS5c0ZMgQnTx5UsHBwZo2bZoiIiKuO88rCGYAAMDSkpOTlZSU5DA2ceJExcfHX/faffv2afTo0dUemzx5siQ5PP/l6ekpSSosLLzmPbOzs7Vu3Tp9%2BumnDuMeHh56%2BOGHNXnyZPn4%2BOijjz7S008/rXXr1ikoKOi6c5UIZgAAwCS11ZhFR0erR48eDmP%2B/v41ujYqKkqHDx%2Bu9lh6errefvttFRcX2x/2v7KFWa9evWvec9WqVQoPD6/SqF151uyKp59%2BWqtXr9aWLVs0cuTIGs2XYAYAAExRW8EsICCgRtuWN6pFixZyc3NTVlaW2rdvL0k6cuSI3Nzc1Lx582te98UXX%2Bipp56qMv6nP/1Jffr00YMPPmgfKy0tlbu7e43nxMP/AADgruTp6am%2BffsqMTFR%2Bfn5ys/PV2Jiovr37y8PD49qr/nhhx905MgRdezYscqxb7/9VrNnz1ZeXp5KS0uVlJSkgoIC9e7du8ZzIpgBAABTVFbWzkdtmjVrlpo3b64BAwbol7/8pe6//369/PLL9uNPPPGE3nvvPfvXJ06ckCQFBgZWudcbb7yhZs2aadCgQYqKilJqaqo%2B%2BOAD%2Bfr61ng%2BLjabzWbg57m7XfXmcQAMCA%2BX9uyRIiKktDRnzwa4/TnpV/tjj9XOfb/%2Bunbua0U8YwYAAExxO77BrNUQzAAAgCkIZsbxjBkAAIBF0JgBAABT0JgZR2MGAABgETRmAADAFDRmxhHMAACAKQhmxrGVCQAAYBE0ZgAAwBQ0ZsbRmAEAAFgEjRkAADAFjZlxBDMAAGAKgplxbGUCAABYBI0ZAAAwBY2ZcTRmAAAAFkFjBgAATEFjZhzBDAAAmIJgZhxbmQAAABZBYwYAAExBY2YcjRkAAIBF0JgBAABT0JgZRzADAACmIJgZx1YmAACARdCYAQAAU9CYGUdjBgAAYBE0ZgAAwBQ0ZsYRzAAAgCkIZsaxlQkAAGARNGYAAMAUNGbGEcwAAIApCGbGsZUJAABgETRmAADAFDRmxtGYAQAAWASNGQAAMAWNmXEEMwAAYAqCmXFsZQIAAFgEjRkAADAFjZlxNGYAAAAWQWMGAABMQWNmHMEMAACYgmBmHFuZAAAAFkFjBgAATEFjZhyNGQAAgEUQzAAAgCkqK2vn41YoLi5WdHS0Vq9e/bPn7du3T08%2B%2BaTCw8PVo0cPffrppw7H16xZo969eyssLEy//vWvlZaWdkPzIJgBAABT3K7BLDMzUyNGjNDevXt/9rwff/xR48aN0%2BDBg7Vz507Nnj1bb7zxhv79739LklJSUpSQkKA5c%2BZo586dGjhwoCZMmKDi4uIaz4VgBgAA7lrbt2/XmDFj9Ktf/UpNmjT52XO/%2BOIL%2Bfr6asSIEapTp44eeeQRDRgwQB999JEk6dNPP9UTTzyhDh06yM3NTWPHjlXDhg21YcOGGs%2BHh/8BAIApaqvdys3NVV5ensOYv7%2B/AgICrnttSUmJzpw5U%2B0xf39/tW7dWl9%2B%2BaXc3d31wQcf/Oy9MjMzFRoa6jDWsmVLrVy5UpKUlZWlIUOGVDmekZFx3XleQTAzwmZz9gyAO0Jubq6SFyxQ9Oef1%2BgfWgDWVFu/FhcsSFZSUpLD2MSJExUfH3/da/ft26fRo0dXe%2Bydd95Rr169ajyPwsJCeXp6Oox5eHioqKioRsdrgmAGwOny8vKUlJSkHj16EMwAVBEdHa0ePXo4jPn7%2B9fo2qioKB0%2BfNiUeXh6eurixYsOYyUlJfL29rYfLykpqXK8YcOGNf4eBDMAAGBpAQEBlvifttDQUH399dcOY1lZWQoJCZEkhYSEKDMzs8rxbt261fh78PA/AABADfTu3Vtnz57V0qVLVVZWph07dmj9%2BvX258qGDh2q9evXa8eOHSorK9PSpUt17tw59e7du8bfg2AGAABwDU888YTee%2B89SVLDhg31l7/8RZ9//rmioqI0c%2BZMzZw5U507d5YkPfLII5o1a5ZeeeUVderUSZ999pkWL14sX1/fGn8/F5uNJ9gBOFdubq6Sk5MVHR1tie0KAHAWghkAAIBFsJUJAABgEQQzAAAAiyCYAQAAWATBDAAAwCIIZgAAABZBMAMAALAIghkAAIBFEMwAAAAsgmAGAABgEQQzAAAAiyCYAQAAWEQdZ08AwN1l586d1z2nY8eOt2AmAGA9/BFzALdU69atJUkuLi72MR8fH128eFGVlZXy9fXV9u3bnTU9AHAqGjMAt1RGRoYkacmSJfr22281c%2BZM1a9fX0VFRZozZ458fHycPEMAcB4aMwBO8eijj2rz5s3y8PCwj126dEndunVTSkqKE2cGAM7Dw/8AnKKyslLnzp1zGDtx4oTuueceJ80IAJyPrUwATjFo0CA9/fTTiomJUePGjZWdna33339fv/nNb5w9NQBwGrYyAThFeXm53nnnHa1bt05nzpxR48aN9eSTTyo2NtbhhQEAcDchmAEAAFgEW5kAbqn//u//1rhx45SUlHTNcyZOnHgLZwQA1kEwA3BL7dy5U%2BPGjbvmKy/ZxgRwN2MrE4BlXWnXAOBuQTADYFkRERHas2ePs6cBALcM72MGwLL4/0YAdxuCGQDL4nkzAHcbghkAAIBFEMwAAAAsgmAGAABgEQQzAJbFw/8A7jYEMwCW9eSTTzp7CgBwS/E%2BZgCcYtSoUdW%2B6tLNzU333nuvunfvrn79%2BjlhZgDgPDRmAJyiffv2OnTokB566CH169dPYWFhOnz4sO699175%2Bflp9uzZWrZsmbOnCQC3FI0ZAKcYPny4nn32WUVGRtrH9u3bp7feekvLly9XRkaGJk%2BerI0bNzpxlgBwa9GYAXCKb7/9VhEREQ5jDz30kNLT0yVJrVu3Vl5enjOmBgBOQzAD4BRBQUFatWqVw9j69evVpEkTSdLBgwfl7%2B/vjKkBgNOwlQnAKb755htNmDBBbdq0UdOmTXXq1CllZGRo/vz58vPz0/DhwzVjxgwNHTrU2VMFgFuGYAbAaU6cOKH169fr9OnTatq0qQYNGqTAwECdPn1aP/zwg9q0aePsKQLALUUwAwAAsIg6zp4AgLtTZmam3nzzTR07dkyVlZUOxzZt2uSkWQGAcxHMADjFyy%2B/LE9PT40bN0516vBPEQBIBDMATnL48GFt3bpV9erVc/ZUAMAyeLsMAE4REBCg0tJSZ08DACyFh/8BOMXy5cv12WefafTo0fLz83M41rFjRyfNCgCci2AGwClat25d7biLi4sOHTp0i2cDANZAMAMAALAIHv4HcEudPn1ajRo10qlTp655zpU/ywQAdxsaMwC3VEREhPbs2aPWrVvLxcVFV/4JuvI5W5kA7mYEMwC3VE5Ojho3bqyTJ09e85ymTZvewhkBgHUQzAA4xYQJE/Tuu%2B9WGR85cqSWL1/uhBkBgPPxjBmAW%2BbEiRNau3atJOlf//qXkpKSHI4XFBTo8OHDzpgaAFgCwQzALdOkSRNlZmYqPz9fFRUVSklJcTju7u6uWbNmOWl2AOB8bGUCcIqZM2fqD3/4g7OnAQCWQjAD4DQHDhxQu3btdPHiRb333nu69957NWbMGP6oOYC7Fv/6AXCKd999V%2B%2B//752796thIQEHThwQK6urjp9%2BrRmzJjh7OkBgFPQmAFwiieeeELz5s3TAw88oI4dOyo5OVn%2B/v4aOHCgvv76a2dPDwCcgsYMgFPk5uaqdevW2r59u%2BrXr2//25nFxcVOnhkAOI%2BrsycA4O4UGBionTt3au3atXrkkUckSX//%2B98VFBTk5JkBgPOwlQnAKTZu3Kjnn39eHh4e%2Bvjjj3XmzBmNGzdOCxYs0OOPP%2B7s6QGAUxDMADjNpUuXJF1%2B/7KCggIVFRUpICDAybMCAOchmAG4pXbv3q0OHTpo586d1zynY8eOt3BGAGAdBDMAt1RERIT27Nljf9j/p1xcXHTo0KFbPCsAsAaCGYBb6uTJk3JxcdG1/ulxcXFRkyZNbvGsAMAaCGYAbqnWrVvLxcXlZ8%2BhMQNwtyKYAbilTp48ed1zmjZtegtmAgDWQzADAACwCN5gFgAAwCIIZgAAABZBMAMAALAIghkAAIBFEMwAAAAsgmAGAABgEQQzAAAAi/j/DNo%2B2/UedBQAAAAASUVORK5CYII%3D" class="center-img">
    <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAmYAAAIaCAYAAACKzIEwAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4xLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvDW2N/gAAIABJREFUeJzt3XlcVXXi//H3RVEWF0zA3Z8OgnsJolhpk9vD0dxKk3Kv1FETsbHNtKwcTQunSdFy1BknbfpibqOTk41l2mK4gI4bCpqFooCoKft2f3/05X69gYmeg/eQr%2BfjcR/J5yx8OD2Sd%2B/Puefa7Ha7XQAAAHA5N1dPAAAAAD8hmAEAAFgEwQwAAMAiCGYAAAAWQTADAACwCIIZAACARRDMAAAALIJgBgAAYBEEMwAAAIuo6uoJALg5CQkJWrdunXbv3q3U1FTl5eXprrvuUmBgoB588EENHTpUHh4erp4mAOAW2PhIJqDyWLRokd59910VFxerRo0aatq0qdzd3ZWenq6UlBRJUoMGDbRkyRK1bdvWxbMFANwsghlQSaxfv14vvfSSvLy89MYbb6h3796qUqWKY/vJkyf10ksv6cCBA6pTp462bt2qu%2B66y4UzBgDcLO4xAyqJ9957T5L0/PPP63e/%2B51TKJOkgIAAvfvuu6pbt64uXbqk999/3xXTBAAYQDADKoErV67ohx9%2BkCTdc889193vrrvuUq9evSRJ//3vf2/L3AAA5uHmf6ASqFr1//5T3bFjh9q0aXPdfSMiIjR69GjVrVvXMfbiiy9q48aNmjFjhrp166a3335be/fuVX5%2Bvv7f//t/evjhh/XYY4%2BpevXqZZ5z7969Wr16teLi4nT58mXVqlVLHTp00KhRo3TvvfeWecyVK1f0P//zP9q5c6eSkpKUmZkpT09PNW3aVN27d9fo0aNVu3Ztp2NatmwpSfr66681f/58ffbZZ3Jzc1Pbtm3117/%2BVbNmzdLGjRv1xz/%2BUZ06ddLixYv17bff6urVq2rcuLEeffRRjR07VjabTZ9%2B%2Bqn%2B/ve/69ixYyouLlarVq00adIk/fa3vy0119zcXK1fv17bt2/X8ePHdeXKFVWrVk0NGzZU165d9cQTT6hevXpOx/To0UNnz57V1q1blZGRoRUrVujgwYPKzs5W48aN1bdvXz311FPy9va%2B7r8rAPg57jEDKonHH39ccXFxstlsGjRokIYOHaqQkJBSS5plKQlmjzzyiLZt26bs7GwFBgaqsLBQp06dkiR17NhRy5YtU82aNZ2OjYqK0vLlyyVJtWvXVuPGjZWWlqb09HRJ0rhx4/Tcc885HXP69GmNHTtW586dU9WqVdW0aVN5enrq7Nmzunz5siSpefPmWr9%2BvVNwKQlmISEhio%2BPV1BQkC5evKiwsDAtXLjQ6ef497//rcLCQgUEBCgjI8MxnwkTJshms2nZsmWqVauWmjRpou%2B%2B%2B07Z2dmy2Wz6y1/%2BogceeMDxPS9evKgxY8boxIkTstlsatq0qWrWrKnU1FTHOevWrasNGzaofv36juNKgtkTTzyhVatWqVq1amrWrJl%2B/PFHnT9/XpIUHBysDz74oFz/jgBAkmQHUCkcOXLE3qFDB3tQUJDjFRISYh8/frx92bJl9gMHDtiLiorKPPaFF15wHNO9e3f70aNHHdvi4uLs9913nz0oKMj%2B8ssvOx334Ycf2oOCguyhoaH2f/7zn47x4uJi%2B8cff%2ByYz9q1a52OGzlypD0oKMg%2BbNgwe2pqqtNxGzdutLdq1coeFBRkX7NmjdNxJXNs166dfc%2BePXa73W4vKiqyX7p0qdTP8fjjj9vT0tIc%2B7z44ov2oKAge6tWrewtW7a0r1y50nE9Ll68aB88eLA9KCjIPnLkyDKvTe/eve3fffed07Zdu3bZ77nnHntQUJB9/vz5Ttu6d%2B/umMuLL75ov3LliuNnXLNmjWPbf/7znzL/nQBAWbjHDKgk2rRpo48%2B%2BkgdO3Z0jGVmZmrnzp1auHChhg0bpq5du%2Brtt99WTk5Omedwc3PT0qVL1bp1a8dYcHCwFixYIEn66KOPlJqaKknKz8/X4sWLJUnz5s3TwIEDHcfYbDb169fP0ZQtXrxYhYWFkqSMjAwlJiZKkubMmSN/f3%2Bn4wYPHqzOnTtLko4fP17mPPv27atOnTo55uzj4%2BO0vWrVqvrTn/4kPz8/xz4TJkyQJBUXF2vQoEF68skn5eb2019xderU0ejRoyVJR48edZynsLBQ%2B/btk81m04wZM9SsWTOn79OtWzf169dPknTixIky59qqVSvNmzfP0TTabDaNGDHC0f7t37%2B/zOMAoCwEM6ASadGihf7xj39o06ZNmjJlioKDg%2BXu7u7YnpGRoffee08DBw50LKddq0uXLmrVqlWp8a5du6px48YqLi7Wjh07JEnx8fG6cOGCvL291bNnzzLnM3DgQLm5uSk1NdUReOrWratvv/1WBw8eVFBQUKljioqKVKNGDUk/3dtVlmvDZ1latmzptKwoSY0aNXL8uaz7yEoCYmZmpmOsatWq2r59uw4ePKgHH3yw1DF2u11eXl6/ONcHH3xQNput1PhvfvMbSdLVq1d/8WcBgGtx8z9QCbVu3VqtW7dWRESEcnJyFBcXp6%2B%2B%2Bkr//Oc/lZGRoR9%2B%2BEGRkZGKiYlxOu7uu%2B%2B%2B7jlbtmypM2fO6PTp05LkaL0KCgo0YsSI6x5XpUoVFRcX69SpU07n9/Dw0Llz53Tw4EH98MMPSk5O1smTJ3Xs2DFlZ2dL%2BqndKktJE3Y9DRo0KDVWrVo1x5/r1KlTavu1b6D4uerVqysjI0MHDhzQ6dOndebMGZ06dUrHjh3Tjz/%2B%2BItzvbYRvFbJpy8UFRVd/wcBgJ8hmAGVnKenp%2B6//37df//9ioyM1EsvvaSPP/5YBw4c0JEjR5w%2BAeDn74K8VkkzdOXKFUn/1/Tk5%2BcrLi7uhvMoOU6STp06pTfffFM7d%2B50CjQ1atRQaGio0tLSlJCQcN1z3egjpTw9PX9xe8kSZnmkp6drwYIF%2BuSTT1RQUOD0Pdq3b6%2BioqJfXI68NhCWxc77qwDcBIIZUAm88sor%2Bvbbb/Xwww9r0qRJ193Pw8NDr7/%2Buj799FMVFBTou%2B%2B%2BcwpmJU1VWUqW%2BEoes1ESftq2basNGzaUe64ZGRkaOXKkMjIy1LBhQw0bNkxt2rTRb37zGzVu3Fg2m03Tp0//xWB2u%2BTl5WnMmDE6efKkfHx89Pjjj6tdu3YKCAhQ06ZNVaVKFb399tvcJwbgtiGYAZVAXl6evv/%2Be23fvv0Xg5n0Uyvl7e2ty5cvl/pIppLlybKUBKUWLVpI%2BulxFtJPj74oLCwscynQbrcrNjZW9evXV8OGDVWtWjWtX79eGRkZ8vHx0fr168v8WKiSNxi42vbt23Xy5ElVrVpVMTExpW7%2Bl1TmvXoAUFG4%2BR%2BoBEreEXn48OEbtldfffWVLl%2B%2BLB8fn1KfErBr1y7Hs7mutWPHDp07d07VqlVTjx49JEmdOnVSzZo1lZWVdd3vuWXLFo0ZM0Z9%2B/Z1BJgzZ85Ikho2bFhmKEtKStKBAwckuf7%2Bq5K5ent7lxnKLly4oC%2B%2B%2BEKS6%2BcK4M5AMAMqgfvvv199%2BvSRJM2aNUtz5851hIoSeXl5Wr9%2BvaZNmyZJioyMLPXU%2BezsbE2ePFnnzp1zjMXGxmrGjBmSfno4a8ljH7y8vByPoJg7d67Wr1/vdL/Y9u3bNXv2bEk/Pd6iadOmkv7v3YgJCQnatm2bY3%2B73a5du3Zp3Lhxjnu5rvdYj9ulZK4//vij/v73vzvdD3bgwAE98cQTjgfiunquAO4MLGUClURUVJS8vLy0adMmvf/%2B%2B3r//ffVsGFD1a1bV3l5eTp9%2BrTy8/Pl7u6u6dOna/jw4aXO0axZMx07dky9evVSUFCQsrOzHe/C7N%2B/v37/%2B9877T9%2B/HglJydr7dq1eumll/TWW2%2BpcePGSk1NVVpamqSfntL/xz/%2B0XHM0KFD9Y9//EPff/%2B9pk6dqkaNGqlOnTo6d%2B6cMjIy5O7urs6dO2vPnj0uX9Ls0aOHgoODFR8fr3nz5mn58uWqV6%2Be0tPTlZqaKpvNpvvuu0/ffPON0tLSZLfby3w0BgCYhWAGVBLVqlXT/PnzNWLECG3dulWxsbFKTU1VQkKCPD091bx5c3Xt2lVDhw51NEE/1759e0VFRWnRokXav3%2B/qlatqs6dO%2Bvxxx93PEj1WjabTXPmzFGfPn30P//zPzpw4ICOHTum6tWrq0OHDurfv7/Cw8Od3plYo0YNrVu3TsuXL9eOHTt05swZXbhwQfXr19eDDz6oMWPGyMvLS7169VJCQoJSUlLUsGHDCrtuv6RKlSpatWqVVq9erY8//ljJyck6ceKE/Pz81K9fP40YMUJt27ZVWFiYLl%2B%2BrLi4uBs%2BYw0AjOCzMoE7QMlnTA4YMEBRUVGung4A4Dq4xwwAAMAiCGYAAAAWQTADAACwCIIZAAC4I1y8eFG9e/dWbGzsdffZuXOnBgwYoA4dOqhv377asWOH0/bly5frgQceUIcOHTRq1CidOnXK1DkSzIA7wPz583X8%2BHFu/Adwx9q/f7/Cw8P1ww8/XHef06dPKyIiQpGRkdq3b58iIiI0bdo0x6N9Nm7cqNWrV2vlypWKjY1V27ZtNXXqVFM/E5dgBgAAftU2btyoZ599Vs8888wN9wsNDVWvXr1UtWpV9evXT506dVJMTIwkae3atRo%2BfLgCAwNVvXp1TZ8%2BXSkpKb/YwN0sghkAALC0tLQ0HTlyxOlV8pDr8ujatav%2B85//lPm8xmslJSUpKCjIaaxFixaOzxL%2B%2BXZ3d3c1a9bMsd0MPGDWCJ4ADpijeXMpMVEKDJS%2B%2B87VswEqP1c9orSCfi/GLFqk6Ohop7EpU6YoIiKiXMf7%2BfmVa7%2BsrCx5eno6jXl4eCg7O7tc281AMAPgej4%2BUpUqP/0TAH4mPDxcPXr0cBorb9i6GZ6ensrNzXUay83NdXzu8I22m4FgBgAAzOFWMXdI%2Bfv7y9/fv0LOfa2goCAdOXLEaSwpKUnt2rWTJAUGBioxMVHdu3eXJBUUFOj06dOllj%2BN4B4zAABgDje3inndJgMHDtSePXu0detWFRYWauvWrdqzZ48GDRokSRoyZIjWrFmjhIQE5eXlaeHChfL19VVoaKhpcyCYAQCAO1ZwcLA2b94sSQoICNCSJUu0bNkyderUSUuXLtXixYvVvHlzSdLQoUM1duxYPf300%2BrSpYuOHj2qZcuWyd3d3bT58CHmRnDzP2CO4GApLk4KCZHi4109G6Dyc9Wv9urVK%2Ba8eXkVc14LojEDAACwCG7%2BBwAA5riN94P9WhHMAACAOQhmhnEFAQAALILGDAAAmIPGzDCuIAAAgEXQmAEAAHPQmBlGMAMAAOYgmBnGFQQAALAIGjMAAGAOGjPDuIIAAAAWQWMGAADMQWNmGMEMAACYg2BmGFcQAADAImjMAACAOWjMDOMKAgAAWASNGQAAMAeNmWEEMwAAYA6CmWFcQQAAAIugMQMAAOagMTOMKwgAAGARNGYAAMAcNGaGEcwAAIA5CGaGcQUBAAAsgsYMAACYg8bMMK4gAACARdCYAQAAc9CYGUYwAwAA5iCYGcYVBAAAsAgaMwAAYA4aM8O4ggAAABZBYwYAAMxBY2YYwQwAAJiDYGYYVxAAAMAiaMwAAIA5aMwM4woCAABYBI0ZAAAwB42ZYQQzAABgDoKZYVxBAAAAi6AxAwAA5qAxM4xgBgAAzEEwM4wrCAAAYBE0ZgAAwBw0ZoYRzAAAwK9aRkaGXn75Ze3Zs0dVqlTRwIED9cILL6hqVecYNG7cOO3fv99pLDs7W%2BHh4Xr99dd14cIF3X///fLy8nJsr1Onjj7//HPT5kowAwAA5rBoYzZt2jTVq1dPX375pS5cuKBJkyZp1apVGjdunNN%2BK1ascPp63bp1io6O1pQpUyRJhw4dUqNGjUwNYj9nzSsIAAAqHze3inkZ8P3332vPnj167rnn5OnpqSZNmmjy5Mn64IMPfvG4U6dOac6cOYqKipK/v7%2Bkn4JZu3btDM3nRmjMAACApaWlpSk9Pd1pzM/PzxGYfkliYqJ8fHxUr149x1hAQIBSUlJ05coV1apVq8zjXnvtNQ0ePFihoaGOsUOHDunHH39U//79deHCBbVv314vvPCCWrRocYs/WWkEMwAAYI4KWsqMiYlRdHS009iUKVMUERFxw2OzsrLk6enpNFbydXZ2dpnBbN%2B%2BfTp48KCioqKcxmvVqqUWLVpo/Pjxqlatmt555x098cQT2rp1q2rWrHmzP1aZCGYAAMDSwsPD1aNHD6cxPz%2B/ch3r5eWlnJwcp7GSr729vcs8JiYmRn379i31PRYuXOj09YwZM7R%2B/Xrt27dP3bt3L9d8boRgBgAAzFFBjZm/v3%2B5li3LEhgYqMuXL%2BvChQvy9fWVJJ08eVL169cvs%2BUqLCzUZ599piVLljiNZ2ZmasmSJRo5cqQaNWokSSoqKlJhYaE8PDxuaW5l4eZ/AABgDgve/N%2BsWTN17NhR8%2BbNU2ZmppKTk7V06VINHTq0zP2PHz%2BuvLw8hYSEOI3XqFFD33zzjRYsWKCrV68qKytLc%2BbMUePGjZ3uQzOKYAYAAH7VFi1apMLCQvXs2VPDhg1Tt27dNHnyZElScHCwNm/e7Ng3OTlZtWvXVvXq1UudZ%2BnSpSouLlavXr3UrVs3paena/ny5XJ3dzdtrja73W437Wx3GpvN1TMAfh2Cg6W4OCkkRIqPd/VsgMrPVb/a%2B/SpmPNu21Yx57UgGjMAAACL4OZ/AABgDos%2B%2Bb8yIZgBAABzEMwM4woCAABYBI0ZAAAwB42ZYVxBAAAAi6AxAwAA5qAxM4xgBgAAzEEwM4wrCAAAYBE0ZgAAwBw0ZoZxBQEAACyCxgwAAJiDxswwghkAADAHwcwwriAAAIBF0JgBAABz0JgZxhUEAACwCBozAABgDhozwwhmAADAHAQzw7iCAAAAFkFjBgAAzEFjZhhXEAAAwCJozAAAgDlozAwjmAEAAHMQzAzjCgIAAFgEjRkAADAHjZlhXEEAAACLoDEDAADmoDEzjGAGAADMQTAzjCsIAABgETRmAADAHDRmhhHMAACAOQhmhnEFAQAALILGDAAAmIPGzDCuIAAAgEXQmAEAAHPQmBlGMAMAAOYgmBnGFQQAALAIGjMAAGAOGjPDuIIAAAAWQWMGAADMQWNmGMEMAACYg2BmGFcQAADAImjMAACAOWjMDOMKAgCAX7WMjAxNnjxZoaGhCgsL09y5c1VYWFjmvuPGjVP79u0VHBzseO3atUuSVFRUpAULFui%2B%2B%2B5TcHCwJk2apLS0NFPnSjADAADmcHOrmJdB06ZNk5eXl7788kutW7dOu3fv1qpVq8rc9/Dhw1q5cqXi4%2BMdrwceeECS9O677%2Brrr7/W%2BvXr9eWXX8rDw0OzZs0yPL9rEcwAAIA5LBjMvv/%2Be%2B3Zs0fPPfecPD091aRJE02ePFkffPBBqX2Tk5P1448/qk2bNmWe66OPPtL48ePVoEED1ahRQzNnztSuXbuUnJxsaI7X4h4zAABgaWlpaUpPT3ca8/Pzk7%2B//w2PTUxMlI%2BPj%2BrVq%2BcYCwgIUEpKiq5cuaJatWo5xg8dOiRvb28988wzOnTokHx9fTV27FgNHTpUV69e1fnz5xUUFOTY39fXV7Vr19bx48fVpEkTE35SghkAADBLBd38HxMTo%2BjoaKexKVOmKCIi4obHZmVlydPT02ms5Ovs7GynYJafn68OHTromWeeUWBgoGJjYxURESFvb28FBwdLkry8vJzO5eHhoaysrFv6ucpCMAMAAJYWHh6uHj16OI35%2BfmV61gvLy/l5OQ4jZV87e3t7TQ%2BePBgDR482PF1165dNXjwYP373//Wfffd53Rsidzc3FLnMYJgBgAAzFFBjZm/v3%2B5li3LEhgYqMuXL%2BvChQvy9fWVJJ08eVL169dXzZo1nfZdt26dvL291bdvX8dYfn6%2Bqlevrtq1a6tevXpKSkpyLGemp6fr8uXLTsubRnHzPwAAMIcFb/5v1qyZOnbsqHnz5ikzM1PJyclaunSphg4dWmrfzMxMzZkzR0ePHlVxcbG%2B%2BOIL/etf/1J4eLgk6ZFHHtG7776r5ORkZWZmat68eercubOaNm1qaI7XojEDAAC/aosWLdLrr7%2Bunj17ys3NTYMHD9bkyZMlScHBwXrttdc0cOBAjRkzRtnZ2ZoyZYoyMjLUpEkTLViwQKGhoZKkp59%2BWoWFhRoxYoSysrIUFhamP//5z6bO1Wa32%2B2mnvFOYrO5egbAr0NwsBQXJ4WESPHxrp4NUPm56lf7G29UzHlnzKiY81oQS5kAAAAWwVImAAAwB5%2BVaRjBDAAAmINgZhhXEAAAwCJozAAAgDlozAzjCgIAAFgEjRkAADAHjZlhBDMAAGAOgplhXEEAAACLoDEDAADmoDEzjCsIAABgETRmAADAHDRmhhHMAACAOQhmhnEFAQAALILGDAAAmIPGzDCuIAAAgEXQmAEAAHPQmBlGMAMAAOYgmBnGFQQAALAIGjMAAGAOGjPDuIIAAAAWQWMGAADMQWNmGMEMAACYg2BmGFcQAADAImjMAACAOWjMDCOYAQAAcxDMDOMKAgAAWASNGQAAMAeNmWFcQQAAAIugMQMAAOagMTOMYAYAAMxBMDOMKwgAAGARNGYAAMAcNGaGcQUBAAAsgsYMAACYg8bMMIIZAAAwB8HMMK4gAACARdCYAQAAc9CYGcYVBAAAsAgaMwAAYA4aM8MIZgAAwBwEM8O4ggAAABZBYwYAAMxBY2YYwQwAAPyqZWRk6OWXX9aePXtUpUoVDRw4UC%2B88IKqVi0dgz788EOtWrVKaWlp8vf31%2BjRozVixAhJUnFxsTp27Ci73S6bzeY45uuvv5aXl5cpcyWYAQAAc1i0MZs2bZrq1aunL7/8UhcuXNCkSZO0atUqjRs3zmm/7du3609/%2BpOWL1%2Bue%2B65RwcOHNCECRPk6%2BurPn36KCkpSQUFBYqLi1O1atUqZK4EMwAAYI4KCmZpaWlKT093GvPz85O/v/8Nj/3%2B%2B%2B%2B1Z88e7dq1S56enmrSpIkmT56st956q1QwS01N1fjx49WhQwdJUnBwsMLCwrR371716dNHhw4dUsuWLSsslEkEMwAAYHExMTGKjo52GpsyZYoiIiJueGxiYqJ8fHxUr149x1hAQIBSUlJ05coV1apVyzFesmRZIiMjQ3v37tWMGTMkSYcOHVJeXp6GDBmis2fPKiAgQNOnT1dISIiRH88JwQwAAJijghqz8PBw9ejRw2nMz8%2BvXMdmZWXJ09PTaazk6%2BzsbKdgdq309HT9/ve/V7t27dS/f39JkoeHh%2B6%2B%2B25FRkaqdu3a%2BuCDD/TUU09p8%2BbNatKkyc3%2BWGUimAEAAEvz9/cv17JlWby8vJSTk%2BM0VvK1t7d3mcccOHBAkZGRCg0N1RtvvOF4k8CLL77otN9TTz2lDRs2aOfOnRo5cuQtze/nrHmXHgAAqHzc3CrmZUBgYKAuX76sCxcuOMZOnjyp%2BvXrq2bNmqX2X7duncaOHasxY8Zo4cKFTveTvf322zp69KjT/vn5%2BapevbqhOV6LYAYAAMxhwWDWrFkzdezYUfPmzVNmZqaSk5O1dOlSDR06tNS%2B27Zt06uvvqrFixfrySefLLX9xIkTmjt3rtLT05Wfn6/o6GhlZmaqd%2B/ehuZ4LYIZAAD4VVu0aJEKCwvVs2dPDRs2TN26ddPkyZMl/fTOy82bN0uSoqOjVVRUpKlTpyo4ONjxeuWVVyRJb7zxhpo2bapBgwYpLCxMe/bs0d/%2B9jf5%2BPiYNleb3W63m3a2O801D5cDYEBwsBQXJ4WESPHxrp4NUPm56lf77t0Vc957762Y81oQjRkAAIBF8K5MAABgDos%2B%2Bb8yIZgBAABzEMwM4woCAABYBI0ZAAAwB42ZYVxBAAAAi6AxAwAA5qAxM4xgBgAAzEEwM4wrCAAAYBE0ZgAAwBw0ZoZxBQEAACyCxgwAAJiDxswwghkAADAHwcwwriAAAIBF0JgBAABz0JgZRjADAADmIJgZxhUEAACwCBozAABgDhozw7iCAAAAFkFjBgAAzEFjZhjBDAAAmINgZhhXEAAAwCJozAAAgDlozAzjCgIAAFgEjRkAADAHjZlhBDMAAGAOgplhXEEAAACLoDEDAADmoDEzjCsIAABgETRmAADAHDRmhhHMAACAOQhmhnEFAQAALILGDAAAmIPGzDCuIAAAgEXQmAEAAHPQmBlGMAMAAOYgmBnGFQQAALAIGjMAAGAOGjPDuIIAAAAWQWMGAADMQWNmGMEMAACYg2BmGFcQAADAIiq8MSsqKlJKSoqaNGlS0d8KAAC4Eo2ZYTd1Bc%2BcOaOWLVvqzJkzCg4O1r59%2B254zDPPPKNNmzZJklJSUhQcHKyUlJRbm%2B1NGDdunN57773rbm/ZsqViY2MrfB4AAMC1MjIyNHnyZIWGhiosLExz585VYWFhmfvu3LlTAwYMUIcOHdS3b1/t2LHDafvy5cv1wAMPqEOHDho1apROnTpl6lxvOdrGx8crNDT0hvtdunTJ8eeGDRsqPj5eDRs2vNVvW24rVqzQxIkTK/z7AACA/%2BXmVjEvg6ZNmyYvLy99%2BeWXWrdunXbv3q1Vq1aV2u/06dOKiIhQZGSk9u3bp4iICE2bNk2pqamSpI0bN2r16tVauXKlYmNj1bZtW02dOlV2u93wHEvc8lJmy5Yt9f777yssLEzbtm3TokWLdP78efn7%2B2vAgAGaPHmyZs6cqX379ik%2BPl5HjhzRrFmz1LNnT3322Wdq3LixWrZsqVmzZmnNmjVKS0tTy5Yt9dprr6lly5aSpG%2B%2B%2BUZvvvmmfvjhBwUFBaljx47673//q9WrV99wfqNGjVLnzp0VERGhgoICRUVFadOmTbLZbBo3btxN/7xpaWlKT093GvNr3lz%2BPj43fS4AP9OqlfM/Ady6%2BHjXfe8KWsos83ewn5/8/f1veOz333%2BvPXv2aNeuXfL09FSTJk00efJkvfXWW6XywMaNGxUaGqpevXorY2TZAAAc30lEQVRJkvr166cNGzYoJiZGU6dO1dq1azV8%2BHAFBgZKkqZPn661a9cqNjZWXbp0MeVnNXyPWW5urp577jktX75cYWFhOnr0qEaMGKGuXbtq7ty5%2BuGHHxwB6cyZM6WO//jjj7VmzRp5eHho6tSpevPNN7Vy5UqdOXNGEydO1MyZMzVkyBAdOHBAEydOVOvWrW96jkuXLtUXX3yhdevWqW7dunr11Vdv%2BhwxMTGKjo52GpsSGamIyMibPheA6/jHP1w9A6Dys9lcPQPTlfk7eMoURURE3PDYxMRE%2Bfj4qF69eo6xgIAApaSk6MqVK6pVq5ZjPCkpSUFBQU7Ht2jRQgkJCY7t48ePd2xzd3dXs2bNlJCQYJ1gJkkeHh5at26diouLFRISov3798utnKl51KhR8vPzkyT17dtXy5YtkyRt2bJFrVu3Vnh4uCQpNDRUw4YN06FDh256fv/85z81ceJExxsQZs2apc2bN9/UOcLDw9WjRw%2BnMb8BA6S///2m5wPgZ1q1%2BimUDR8u/e9fgAAqH7sqJhSW%2BTv4f7PDjWRlZcnT09NprOTr7Oxsp2BW1r4eHh7Kzs4u13YzGA5mHh4e%2BvDDD7V06VJNnz5dmZmZ6tOnj2bNmqXatWvf8HhfX9//m0zVqo512nPnzqlRo0ZO%2BzZp0uSWgllaWpoaNGjg%2BLpWrVrlmtu1/P39S1em331303MB8AsSEly7DAPAksr8HVxOXl5eysnJcRor%2Bdrb29tp3NPTU7m5uU5jubm5jv1utN0MhheDMzMzlZaWpoULF%2Bqbb75RTEyMDh8%2B/IvviCyPRo0alXr35q2%2Bm7N%2B/fpKTk52fJ2dna2rV68amh8AAHBWXFwxLyMCAwN1%2BfJlXbhwwTF28uRJ1a9fXzVr1nTaNygoSImJiU5jSUlJjnvKAgMDnbYXFBTo9OnTpZY/jTAczLKysjR%2B/Hht2bJFdrtd/v7%2BcnNzU506dSRJ1apVu6UQNGjQIB07dkybNm1SUVGRDh48qLVr197SHB999FGtWLFCJ0%2BeVF5enubPn6%2BioqJbOhcAACibFYNZs2bN1LFjR82bN0%2BZmZlKTk7W0qVLNXTo0FL7Dhw4UHv27NHWrVtVWFiorVu3as%2BePRo0aJAkaciQIVqzZo0SEhKUl5enhQsXytfXt1xPqSgvw8GsXr16WrRokZYvX66QkBD1799fXbp00dixYyVJgwcP1vr16zV8%2BPCbOm/9%2BvUd5w0NDdWCBQvUtWtXubu73/Qcx48fr4EDB2rkyJHq2rWratasKR/eTQkAwB1h0aJFKiwsVM%2BePTVs2DB169ZNkydPliQFBwc77jsPCAjQkiVLtGzZMnXq1ElLly7V4sWL1bx5c0nS0KFDNXbsWD399NPq0qWLjh49qmXLlt1SNrkem93Mh2%2BY6Ny5c7p06ZLatGnjGJs/f77S09O1cOFCF87sGr/Cd74ALhEcLMXFSSEh3GMGmMFFv9rz8irmvNWrV8x5rciyn51w6dIlDR8%2BXIcPH5YkJSQkaPPmzerevbuLZwYAAFAxKvyzMm9VmzZtNHPmTP3hD39Qenq6fH19NWHCBPXv319z587VunXrrnvs73//e576DwDAbWb0fjBYeCmzUmApEzAHS5mAuVz0qz0rq2LOa%2BLTKCzPskuZAAAAdxrLLmUCAIDKhaVM4whmAADAFAQz41jKBAAAsAgaMwAAYAoaM%2BNozAAAACyCxgwAAJiCxsw4ghkAADAFwcw4ljIBAAAsgsYMAACYgsbMOBozAAAAi6AxAwAApqAxM45gBgAATEEwM46lTAAAAIugMQMAAKagMTOOxgwAAMAiaMwAAIApaMyMI5gBAABTEMyMYykTAADAImjMAACAKWjMjKMxAwAAsAgaMwAAYAoaM%2BMIZgAAwBQEM%2BNYygQAALAIGjMAAGAKGjPjaMwAAAAsgsYMAACYgsbMOIIZAAAwBcHMOJYyAQAALILGDAAAmILGzDgaMwAAAIugMQMAAKagMTOOYAYAAExBMDOOpUwAAACLoDEDAACmoDEzjsYMAADAImjMAACAKWjMjCOYAQAAUxDMjGMpEwAAwCJozAAAgClozIwjmAEAgDtWdna25syZo88//1yFhYXq2bOnZs%2BeLW9v7zL337Ztm5YuXark5GT5%2BPjokUce0eTJk%2BXm9tMiZN%2B%2BfZWSkuL4WpLWrVungICAcs2HYAYAAExRGRuzOXPm6Ny5c9q2bZuKioo0bdo0RUVFafbs2aX2PXz4sJ5//nn9%2Bc9/1m9/%2B1t99913Gj9%2BvLy8vPTkk08qMzNT3333nT777DM1atTolubDPWYAAMAUxcUV80pLS9ORI0ecXmlpaYbnm5OToy1btmjq1Kny8fFR3bp19eyzz2rDhg3Kyckptf/Zs2f12GOPqXv37nJzc1NAQIB69%2B6tvXv3SvopuPn4%2BNxyKJNozAAAgMXFxMQoOjraaWzKlCmKiIi44bG5ublKTU0tc1tOTo4KCgoUFBTkGAsICFBubq5Onz6t1q1bO%2B3fp08f9enTx%2BncX3zxhQYMGCBJOnTokDw9PTVy5EglJiaqUaNGioiIUPfu3cv9sxLMAACAKSpqKTM8PFw9evRwGvPz8yvXsQcPHtTo0aPL3BYZGSlJ8vLycox5enpKkrKysn7xvJmZmYqMjJSHh4fGjh0rSbLZbGrfvr3%2B8Ic/qGHDhvrkk08UERGhNWvWqEOHDuWaL8EMAABYmr%2B/v/z9/W/p2LCwMB0/frzMbUePHtU777yjnJwcx83%2BJUuYNWrUuO45T506palTp6pu3bp6//33HfuOGzfOab%2BBAwfqX//6l7Zt21buYMY9ZgAAwBQVdY9ZRWnevLnc3d2VlJTkGDt58qTc3d3VrFmzMo/ZuXOnHn30UXXr1k0rV65U7dq1HdtWrlyp3bt3O%2B2fn5%2Bv6tWrl3tOBDMAAGCKyhbMPD091bdvX0VFRenixYu6ePGioqKi1L9/f3l4eJTa/8CBA3r66ac1Y8YMvfDCC6pa1Xnh8dy5c3rttdeUnJyswsJCrVu3TvHx8Xr44YfLPSeb3W63G/7J7lQ2m6tnAPw6BAdLcXFSSIgUH%2B/q2QCVn4t%2Btf/nPxVz3t69K%2Ba80k/3ii1YsECff/65CgoK1LNnT7388suO%2B84eeughDRgwQBMnTtTEiRP1xRdfOO5DK9GxY0etWLFC%2Bfn5ioqK0r///W9dvXpVLVq00HPPPaewsLByz4dgZgTBDDAHwQwwl4t%2BtW/bVjHnveaNkL963PwPAABMURkfMGs13GMGAABgETRmAADAFDRmxtGYAQAAWASNGQAAMAWNmXEEMwAAYAqCmXEsZQIAAFgEjRkAADAFjZlxNGYAAAAWQWMGAABMQWNmHMEMAACYgmBmHEuZAAAAFkFjBgAATEFjZhyNGQAAgEXQmAEAAFPQmBlHMAMAAKYgmBnHUiYAAIBF0JgBAABT0JgZR2MGAABgETRmAADAFDRmxhHMAACAKQhmxrGUCQAAYBE0ZgAAwBQ0ZsbRmAEAAFgEjRkAADAFjZlxBDMAAGAKgplxLGUCAABYBI0ZAAAwBY2ZcTRmAAAAFkFjBgAATEFjZhzBDAAAmIJgZhxLmQAAABZBYwYAAExBY2YcjRkAAIBF0JgBAABT0JgZRzADAACmIJgZx1ImAACARdCYAQAAU9CYGUdjBgAAYBE0ZgAAwBQ0ZsYRzAAAgCkIZsaxlAkAAGARBDMAAGCK4uKKeVWk7OxszZgxQ2FhYerYsaOef/55ZWVlXXf/2bNnq127dgoODna8YmJiHNuXL1%2BuBx54QB06dNCoUaN06tSpm5oPwQwAAJiiMgazOXPm6Ny5c9q2bZs%2B/fRTnTt3TlFRUdfd/9ChQ5ozZ47i4%2BMdr/DwcEnSxo0btXr1aq1cuVKxsbFq27atpk6dKrvdXu75EMwAAMAdKScnR1u2bNHUqVPl4%2BOjunXr6tlnn9WGDRuUk5NTav/8/HydOHFC7dq1K/N8a9eu1fDhwxUYGKjq1atr%2BvTpSklJUWxsbLnnxM3/AADAFBXVbqWlpSk9Pd1pzM/PT/7%2B/jc8Njc3V6mpqWVuy8nJUUFBgYKCghxjAQEBys3N1enTp9W6dWun/RMSElRYWKhFixZp//79qlmzpoYMGaJx48bJzc1NSUlJGj9%2BvGN/d3d3NWvWTAkJCerSpUu5flaCGQAAsLSYmBhFR0c7jU2ZMkURERE3PPbgwYMaPXp0mdsiIyMlSV5eXo4xT09PSSrzPrOrV6%2Bqc%2BfOGjVqlP70pz/p2LFjevrpp%2BXm5qZx48YpKyvLcXwJDw8PZWdn33CeJQhmAADAFBXVmIWHh6tHjx5OY35%2BfuU6NiwsTMePHy9z29GjR/XOO%2B8oJydH3t7ekuRYwqxRo0ap/e%2B//37df//9jq/vvvtujRkzRlu3btW4cePk6emp3Nxcp2Nyc3Md5y4PghkAADBFRQUzf3//ci1b3qzmzZvL3d1dSUlJuueeeyRJJ0%2BedCxB/tz27dt14cIFPfbYY46x/Px8eXh4SJICAwOVmJio7t27S5IKCgp0%2BvRpp6XSG%2BHmfwAAcEfy9PRU3759FRUVpYsXL%2BrixYuKiopS//79HWHrWna7XW%2B88YZ2794tu92u%2BPh4vf/%2B%2B453ZQ4ZMkRr1qxRQkKC8vLytHDhQvn6%2Bio0NLTcc6IxAwAApqiMT/6fPXu2FixYoAEDBqigoEA9e/bUyy%2B/7Nj%2B0EMPacCAAZo4caJ69%2B6tGTNm6NVXX1Vqaqp8fX0VERGhQYMGSZKGDh2qq1ev6umnn9bFixfVvn17LVu2TO7u7uWej81%2BMw/XgDObzdUzAH4dgoOluDgpJESKj3f1bIDKz0W/2mfOrJjzzp1bMee1IhozAABgisrYmFkNwQwAAJiCYGYcN/8DAABYBI0ZAAAwBY2ZcTRmAAAAFkFjBgAATEFjZhzBDAAAmIJgZhxLmQAAABZBYwYAAExBY2YcjRkAAIBF0JgBAABT0JgZRzADAACmIJgZx1ImAACARdCYAQAAU9CYGUdjBgAAYBE0ZgAAwBQ0ZsYRzAAAgCkIZsaxlAkAAGARNGYAAMAUNGbG0ZgBAABYBI0ZAAAwBY2ZcQQzAABgCoKZcSxlAgAAWASNGQAAMAWNmXE0ZgAAABZBYwYAAExBY2YcwQwAAJiCYGYcS5kAAAAWQWMGAABMQWNmHI0ZAACARdCYAQAAU9CYGUcwAwAApiCYGcdSJgAAgEXQmAEAAFPQmBlHMAMAAKYgmBnHUiYAAIBF0JgBAABT0JgZR2MGAABgETRmAADAFDRmxhHMAACAKQhmxrGUCQAAYBE0ZgAAwBQ0ZsbRmAEAAFgEjRkAADBFZWzMsrOzNWfOHH3%2B%2BecqLCxUz549NXv2bHl7e5fa95VXXtGWLVucxnJzc3Xfffdp5cqVKi4uVseOHWW322Wz2Rz7fP311/Ly8irXfAhmAADAFJUxmM2ZM0fnzp3Ttm3bVFRUpGnTpikqKkqzZ88ute/rr7%2Bu119/3fH1V199penTp%2BvFF1%2BUJCUlJamgoEBxcXGqVq3aLc2HpUwAAHBHysnJ0ZYtWzR16lT5%2BPiobt26evbZZ7Vhwwbl5OT84rEXL17Us88%2Bq5kzZyowMFCSdOjQIbVs2fKWQ5lEYwYAAExSUY1ZWlqa0tPTncb8/Pzk7%2B9/w2Nzc3OVmppa5racnBwVFBQoKCjIMRYQEKDc3FydPn1arVu3vu55o6Ki1K5dOw0cONAxdujQIeXl5WnIkCE6e/asAgICNH36dIWEhNxwniUIZgAAwNJiYmIUHR3tNDZlyhRFRETc8NiDBw9q9OjRZW6LjIyUJKf7vzw9PSVJWVlZ1z1ncnKyNm/erI8%2B%2Bshp3MPDQ3fffbciIyNVu3ZtffDBB3rqqae0efNmNWnS5IZzlQhmAADAJBXVmIWHh6tHjx5OY35%2BfuU6NiwsTMePHy9z29GjR/XOO%2B8oJyfHcbN/yRJmjRo1rnvO9evXKzg4uFSjVnKvWYmnnnpKGzZs0M6dOzVy5MhyzZdgBgAATFFRwczf379cy5Y3q3nz5nJ3d1dSUpLuueceSdLJkyfl7u6uZs2aXfe4Tz/9VE8%2B%2BWSp8bffflt9%2BvRRmzZtHGP5%2BfmqXr16uefEzf8AAOCO5Onpqb59%2ByoqKkoXL17UxYsXFRUVpf79%2B8vDw6PMYy5duqSTJ0%2BqU6dOpbadOHFCc%2BfOVXp6uvLz8xUdHa3MzEz17t273HMimAEAAFMUF1fMqyLNnj1bzZo104ABA/S73/1OjRs31iuvvOLY/tBDD%2Bm9995zfH3mzBlJUr169Uqd64033lDTpk01aNAghYWFac%2BePfrb3/4mHx%2Bfcs/HZrfb7QZ%2BnjvbNQ%2BPA2BAcLAUFyeFhEjx8a6eDVD5uehX%2B/33V8x5v/66Ys5rRdxjBgAATFEZHzBrNQQzAABgCoKZcdxjBgAAYBE0ZgAAwBQ0ZsbRmAEAAFgEjRkAADAFjZlxBDMAAGAKgplxLGUCAABYBI0ZAAAwBY2ZcTRmAAAAFkFjBgAATEFjZhzBDAAAmIJgZhxLmQAAABZBYwYAAExBY2YcjRkAAIBF0JgBAABT0JgZRzADAACmIJgZx1ImAACARdCYAQAAU9CYGUdjBgAAYBE0ZgAAwBQ0ZsYRzAAAgCkIZsaxlAkAAGARNGYAAMAUNGbG0ZgBAABYBI0ZAAAwBY2ZcQQzAABgCoKZcSxlAgAAWASNGQAAMAWNmXEEMwAAYAqCmXEsZQIAAFgEjRkAADAFjZlxNGYAAAAWQWMGAABMQWNmHMEMAACYgmBmHEuZAAAAFkFjBgAATEFjZhyNGQAAgEXQmAEAAFPQmBlHMAMAAKYgmBnHUiYAAIBF0JgBAABT0JgZR2MGAABgEQQzAABgiuLiinndDjk5OQoPD9eGDRt%2Bcb%2BDBw/q0UcfVXBwsHr06KGPPvrIafvGjRvVu3dvdejQQY888oji4%2BNvah4EMwAAYIrKGswSExM1YsQIHThw4Bf3%2B/HHHzVhwgQNHjxYe/fu1dy5c/XGG2/ov//9ryQpNjZWc%2BbM0fz587V3714NHDhQkyZNUk5OTrnnQjADAAB3rN27d2vMmDF6%2BOGH1bBhw1/c99NPP5WPj49GjBihqlWr6t5779WAAQP0wQcfSJI%2B%2BugjPfTQQ%2BrYsaPc3d01duxY1alTR1u3bi33fLj5HwAAmKKi2q20tDSlp6c7jfn5%2Bcnf3/%2BGx%2Bbm5io1NbXMbX5%2BfmrVqpV27Nih6tWr629/%2B9svnisxMVFBQUFOYy1atNC6deskSUlJSRoyZEip7QkJCTecZwmCmRF2u6tnAPwqpKWlKWbxYoV/8km5/qIFYE0V9Wtx8eIYRUdHO41NmTJFERERNzz24MGDGj16dJnblixZol69epV7HllZWfL09HQa8/DwUHZ2drm2lwfBDIDLpaenKzo6Wj169CCYASglPDxcPXr0cBrz8/Mr17FhYWE6fvy4KfPw9PTU1atXncZyc3Pl7e3t2J6bm1tqe506dcr9PQhmAADA0vz9/S3xP21BQUH6%2BuuvncaSkpIUGBgoSQoMDFRiYmKp7Q888EC5vwc3/wMAAJRD7969deHCBa1atUoFBQX69ttvtWXLFsd9ZUOHDtWWLVv07bffqqCgQKtWrVJGRoZ69%2B5d7u9BMAMAALiOhx56SO%2B9954kqU6dOvrrX/%2BqTz75RGFhYZo1a5ZmzZqlLl26SJLuvfdezZ49W6%2B%2B%2Bqo6d%2B6sjz/%2BWMuXL5ePj0%2B5v5/NbucOdgCulZaWppiYGIWHh1tiuQIAXIVgBgAAYBEsZQIAAFgEwQwAAMAiCGYAAAAWQTADAACwCIIZAACARRDMAAAALIJgBgAAYBEEMwAAAIsgmAEAAFgEwQwAAMAiCGYAAAAWUdXVEwBwZ9m7d%2B8N9%2BnUqdNtmAkAWA8fYg7gtmrVqpUkyWazOcZq166tq1evqri4WD4%2BPtq9e7erpgcALkVjBuC2SkhIkCStXLlSJ06c0KxZs1SzZk1lZ2dr/vz5ql27totnCACuQ2MGwCXuu%2B8%2Bff755/Lw8HCM5eXl6YEHHlBsbKwLZwYArsPN/wBcori4WBkZGU5jZ86cUZUqVVw0IwBwPZYyAbjEoEGD9NRTT2ncuHFq0KCBkpOTtWLFCj322GOunhoAuAxLmQBcorCwUEuWLNHmzZuVmpqqBg0a6NFHH9X48eOd3hgAAHcSghkAAIBFsJQJ4Lb6y1/%2BogkTJig6Ovq6%2B0yZMuU2zggArINgBuC22rt3ryZMmHDdd16yjAngTsZSJgDLKmnXAOBOQTADYFkhISGKi4tz9TQA4LbhOWYALIv/bwRwpyGYAbAs7jcDcKchmAEAAFgEwQwAAMAiCGYAAAAWQTADYFnc/A/gTkMwA2BZjz76qKunAAC3Fc8xA%2BASo0aNKvNdl%2B7u7rrrrrvUvXt39evXzwUzAwDXoTED4BL33HOPjh07pvbt26tfv37q0KGDjh8/rrvuuku%2Bvr6aO3euVq9e7eppAsBtRWMGwCWGDx%2BuP/zhDwoNDXWMHTx4UG%2B99ZbWrFmjhIQERUZGatu2bS6cJQDcXjRmAFzixIkTCgkJcRpr3769jh49Kklq1aqV0tPTXTE1AHAZghkAl2jSpInWr1/vNLZlyxY1bNhQknTkyBH5%2Bfm5YmoA4DIsZQJwiW%2B%2B%2BUaTJk1S69at1ahRI6WkpCghIUGLFi2Sr6%2Bvhg8frpkzZ2ro0KGunioA3DYEMwAuc%2BbMGW3ZskXnz59Xo0aNNGjQINWrV0/nz5/XpUuX1Lp1a1dPEQBuK4IZAACARVR19QQA3JkSExP15ptv6vTp0youLnba9tlnn7loVgDgWgQzAC7xyiuvyNPTUxMmTFDVqvxVBAASwQyAixw/fly7du1SjRo1XD0VALAMHpcBwCX8/f2Vn5/v6mkAgKVw8z8Al1izZo0%2B/vhjjR49Wr6%2Bvk7bOnXq5KJZAYBrEcwAuESrVq3KHLfZbDp27Nhtng0AWAPBDAAAwCK4%2BR/AbXX%2B/HnVr19fKSkp192n5GOZAOBOQ2MG4LYKCQlRXFycWrVqJZvNppK/gkr%2BzFImgDsZwQzAbXXu3Dk1aNBAZ8%2Beve4%2BjRo1uo0zAgDrIJgBcIlJkybp3XffLTU%2BcuRIrVmzxgUzAgDX4x4zALfNmTNntGnTJknSV199pejoaKftmZmZOn78uCumBgCWQDADcNs0bNhQiYmJunjxooqKihQbG%2Bu0vXr16po9e7aLZgcArsdSJgCXmDVrlv74xz%2B6ehoAYCkEMwAuc/jwYbVr105Xr17Ve%2B%2B9p7vuuktjxozhQ80B3LH42w%2BAS7z77rtasWKF9u/frzlz5ujw4cNyc3PT%2BfPnNXPmTFdPDwBcgsYMgEs89NBDWrhwoX7zm9%2BoU6dOiomJkZ%2BfnwYOHKivv/7a1dMDAJegMQPgEmlpaWrVqpV2796tmjVrOj47Mycnx8UzAwDXcXP1BADcmerVq6e9e/dq06ZNuvfeeyVJ//rXv9SkSRMXzwwAXIelTAAusW3bNj3//PPy8PDQhx9%2BqNTUVE2YMEGLFy/Wgw8%2B6OrpAYBLEMwAuExeXp6kn55flpmZqezsbPn7%2B7t4VgDgOgQzALfV/v371bFjR%2B3du/e6%2B3Tq1Ok2zggArINgBuC2CgkJUVxcnONm/5%2Bz2Ww6duzYbZ4VAFgDwQzAbXX27FnZbDZd768em82mhg0b3uZZAYA1EMwA3FatWrWSzWb7xX1ozADcqQhmAG6rs2fP3nCfRo0a3YaZAID1EMwAAAAsggfMAgAAWATBDAAAwCIIZgAAABZBMAMAALAIghkAAIBFEMwAAAAsgmAGAABgEf8fLisB8V8pprYAAAAASUVORK5CYII%3D" class="center-img">
</div>
    <div class="row headerrow highlight">
        <h1>Sample</h1>
    </div>
    <div class="row variablerow">
    <div class="col-md-12" style="overflow:scroll; width: 100%%; overflow-y: hidden;">
        <table border="1" class="dataframe sample">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>listing_id</th>
      <th>date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>109</td>
      <td>2011-08-15</td>
    </tr>
    <tr>
      <th>1</th>
      <td>109</td>
      <td>2016-05-15</td>
    </tr>
    <tr>
      <th>2</th>
      <td>344</td>
      <td>2016-06-14</td>
    </tr>
    <tr>
      <th>3</th>
      <td>344</td>
      <td>2016-12-30</td>
    </tr>
    <tr>
      <th>4</th>
      <td>344</td>
      <td>2018-08-28</td>
    </tr>
  </tbody>
</table>
    </div>
</div>
</div>



# Data Preview in an interactive grid widget

(Only this section does NOT work in **Google Colab**.)

Qgrid is a scrollable grid widget that can be used to edit, sort, and filter DataFrames in Jupyter notebooks. It was developed for use in [Quantopian's hosted research environment](https://www.quantopian.com/notebooks/survey?utm_source=quantopian&amp;utm_medium=web&amp;utm_campaign=qgrid-demo-nb) and also was released as an [open source project on GitHub](https://github.com/quantopian/qgrid).


```python
import qgrid
# set the default max number of rows to 12 so the DataFrame we render with qgrid aren't too tall
qgrid.set_grid_option('maxVisibleRows', 12)
# now render the DataFrame using qgrid
grid=qgrid.show_grid(df, show_toolbar=True)
grid
```


    QgridWidget(grid_options={'fullWidthRows': True, 'syncColumnCellResize': True, 'forceFitColumns': True, 'defau…



```python
#Maybe you did change the df in the grid, such as sort, filter, modification
newdf=grid.get_changed_df()
newdf.head()
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
      <th>listing_id</th>
      <th>date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>109</td>
      <td>2011-08-15</td>
    </tr>
    <tr>
      <th>1</th>
      <td>109</td>
      <td>2016-05-15</td>
    </tr>
    <tr>
      <th>2</th>
      <td>344</td>
      <td>2016-06-14</td>
    </tr>
    <tr>
      <th>3</th>
      <td>344</td>
      <td>2016-12-30</td>
    </tr>
    <tr>
      <th>4</th>
      <td>344</td>
      <td>2018-08-28</td>
    </tr>
  </tbody>
</table>
</div>



Depending on your need, you may assign df=newdf to continue.


```python
#df=newdf
```

# Clean your data 

## Drop multiple columns
Sometimes, not all columns are useful in our analysis. Therefore, the df.drop comes in handy to drop the selected columns as specified by you.


```python
def drop_multiple_col(col_names_list, df): 
    '''
    AIM    -> Drop multiple columns based on their column names 
    
    INPUT  -> List of column names, df
    
    OUTPUT -> updated df with dropped columns 
    ------
    '''
    df.drop(col_names_list, axis=1, inplace=True)
    return df

#drop_multiple_col("",df)
```

## Change dtypes
When a dataset gets larger, we need to convert the dtypes in order to save memory. If you’re interested in learning how to use Pandas to deal with large data, I strongly encourage you to check out this article — Why and How to Use Pandas with Large Data(https://towardsdatascience.com/why-and-how-to-use-pandas-with-large-data-9594dda2ea4c).


```python
def change_dtypes(col_int, col_float, df): 
    '''
    AIM    -> Changing dtypes to save memory
     
    INPUT  -> List of column names (int, float), df
    
    OUTPUT -> updated df with smaller memory  
    ------
    '''
    df[col_int] = df[col_int].astype('int32')
    df[col_float] = df[col_float].astype('float32')
    
#change_dtypes(col_int, col_float, df)
```

## Convert categorical variable to numerical variable
Some machine learning models require variables to be in numerical format. This is when we need to convert categorical variables to numerical variables before feeding them to the models. In terms of data visualization, I’d suggest to retain the categorical variables to have a more explicit interpretation and understanding.


```python
def convert_cat2num(df):
    # Convert categorical variable to numerical variable
    num_encode = {'col_1' : {'YES':1, 'NO':0},
                  'col_2'  : {'WON':1, 'LOSE':0, 'DRAW':0}}  
    df.replace(num_encode, inplace=True)  
    
#convert_cat2num(df)
```

## Check missing data
If you want to check the number of missing data for each column, this is the fastest way to go with. This gives you a better understanding of which columns have higher number of missing data that determine your next action of data cleaning and analysis.

The reference to deal with missing data can be found https://pandas.pydata.org/pandas-docs/stable/missing_data.html


```python
def check_missing_data(df):
    # check for any missing data in the df (display in descending order)
    return df.isnull().sum().sort_values(ascending=False)

check_missing_data(df)
```




    date          0
    listing_id    0
    dtype: int64



## Remove strings in columns
There might be some time when you’d face the new line character or other weird symbols that appear in your columns of strings. This could easily be dealt with using df['col_1'].replace where col_1 is one of the columns in the dataframe df.


```python
def remove_col_str(df):
    # remove a portion of string in a dataframe column - col_1
    df['col_1'].replace('\n', '', regex=True, inplace=True)
    
    # remove all the characters after &# (including &#) for column - col_1
    df['col_1'].replace(' &#.*', '', regex=True, inplace=True)

#remove_col_str(df)
```

## Remove white space in columns
Anything is possible when data is messy. It is not uncommon to see there are some white spaces at the beginning of the strings. Thus this approach is useful when you want to remove white spaces at the beginning of the strings in a column.


```python
def remove_col_white_space(df):
    # remove white space at the beginning of string 
    df[col] = df[col].str.lstrip()
#remove_col_white_space(df)
```

## Concatenate two columns with strings (with condition)
This is helpful when you want to combine two columns with strings conditionally. For instance, you want to concatenate the 1st column with the 2nd column if the strings in the 1st column end with certain letters. The ending letters can also be removed after the concatenation, depending on your needs.


```python
def concat_col_str_condition(df):
    # concat 2 columns with strings if the last 3 letters of the first column are 'pil'
    mask = df['col_1'].str.endswith('pil', na=False)
    col_new = df[mask]['col_1'] + df[mask]['col_2']
    col_new.replace('pil', ' ', regex=True, inplace=True)  # replace the 'pil' with emtpy space
    
#concat_col_str_condition(df)
```

## Convert timestamp(from string to datetime format)
When dealing with time series data, chances are we’ll encounter timestamp column in string format. This means we may have to convert the string format to datetime format — format to be specified based on our requirement — in order to give meaningful analysis and presentation using the data.


```python
def convert_str_datetime(df): 
    '''
    AIM    -> Convert datetime(String) to datetime(format we want)
     
    INPUT  -> df
    
    OUTPUT -> updated df with new datetime format 
    ------
    '''
    df.insert(loc=2, column='timestamp', value=pd.to_datetime(df.transdate, format='%Y-%m-%d %H:%M:%S.%f')) 

#convert_str_datetime(df)
```

## Other Cleaning (Sample only)


```python
# Rename columns
df = df.rename(columns = {'date': 'ds', 'listing_id': 'ts'})

# Group data by number of listings per date
df_processed = df.groupby(by = 'ds').agg({'ts': 'count'})

display(df_processed.index)

# Change index to datetime
df_processed.index = pd.to_datetime(df_processed.index)

# Set frequency of time series
df_processed = df_processed.asfreq(freq='1D')

# Sort the values
df_processed = df_processed.sort_index(ascending = True)
```


    Index(['2009-05-26', '2009-06-01', '2009-06-24', '2009-07-23', '2009-07-29',
           '2009-08-13', '2009-08-23', '2009-08-30', '2009-09-02', '2009-09-03',
           ...
           '2018-11-27', '2018-11-28', '2018-11-29', '2018-11-30', '2018-12-01',
           '2018-12-02', '2018-12-03', '2018-12-04', '2018-12-05', '2018-12-06'],
          dtype='object', name='ds', length=3144)



```python
df_processed=df_processed.dropna()
#df_processed.count()
```


```python
df_processed.index
```

Have you **changed** the df after loading?


```python
#df_processed=df # If you didn't change the df
```

# Data Preview in Chart
- There may appear to be an overall increasing trend. 
- There may appear to be some differences in the variance over time. 
- There may be some seasonality (i.e., cycles) in the data.
- There may be some outliers.


```python
df_processed.plot.line() #line plot (default)
# df_processed.plot.bar() #vertical bar plot
# df_processed.plot.barh() #horizontal bar plot
# df_processed.plot.hist() #histogram
# df_processed.plot.box() #boxplot
# df_processed.plot.kde() #Kernel Density Estimation plot
# df_processed.plot.density() #same as ‘kde’
# df_processed.plot.area() #area plot
# df_processed.plot.pie() #pie plot
# df_processed.plot.scatter() #scatter plot
# df_processed.plot.hexbin() #hexbin plot
```




    <matplotlib.axes._subplots.AxesSubplot at 0x27800762e10>




![png](Time-series%20Preprocessing_files/Time-series%20Preprocessing_39_1.png)


# Deal with stationarity
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


```python
from statsmodels.tsa.stattools import adfuller
def test_stationarity(df_ts):
    """
    Test stationarity using moving average statistics and Dickey-Fuller test
    Source: https://www.analyticsvidhya.com/blog/2016/02/time-series-forecasting-codes-python/
    """
    
    # Determing rolling statistics
    rolmean = df_ts.rolling(window = 12, center = False).mean()
    rolstd = df_ts.rolling(window = 12, center = False).std()
    
    # Plot rolling statistics:
    orig = plt.plot(df_ts, 
                    color = 'blue', 
                    label = 'Original')
    mean = plt.plot(rolmean, 
                    color = 'red', 
                    label = 'Rolling Mean')
    std = plt.plot(rolstd, 
                   color = 'black', 
                   label = 'Rolling Std')
    plt.legend(loc = 'best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.xticks(rotation = 45)
    plt.show(block = False)
    plt.close()
    
    # Perform Dickey-Fuller test:
    # Null Hypothesis (H_0): time series is not stationary
    # Alternate Hypothesis (H_1): time series is stationary
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(df_ts, 
                      autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], 
                         index = ['Test Statistic',
                                  'p-value',
                                  '# Lags Used',
                                  'Number of Observations Used'])
    
    if (dftest[1]>0.05):
        print("The time series is NOT stationary at the p = 0.05 level.")
    else:
        print("The time series is stationary at the p = 0.05 level.")
                    
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
```

Adfuller and most other tsa functions cannot handle missing values.


```python
ts=df_processed['ts']  # You may replace this with your series
test_stationarity(ts)
```


![png](Time-series%20Preprocessing_files/Time-series%20Preprocessing_43_0.png)


    Results of Dickey-Fuller Test:
    The time series is NOT stationary at the p = 0.05 level.
    Test Statistic                   -1.606645
    p-value                           0.480255
    # Lags Used                      29.000000
    Number of Observations Used    3114.000000
    Critical Value (1%)              -3.432452
    Critical Value (5%)              -2.862469
    Critical Value (10%)             -2.567264
    dtype: float64
    

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


```python
def plot_decomposition(df, ts, trend, seasonal, residual):
  """
  Plot time series data
  """
  f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize = (15, 5), sharex = True)

  ax1.plot(df[ts], label = 'Original')
  ax1.legend(loc = 'best')
  ax1.tick_params(axis = 'x', rotation = 45)

  ax2.plot(df[trend], label = 'Trend')
  ax2.legend(loc = 'best')
  ax2.tick_params(axis = 'x', rotation = 45)

  ax3.plot(df[seasonal],label = 'Seasonality')
  ax3.legend(loc = 'best')
  ax3.tick_params(axis = 'x', rotation = 45)

  ax4.plot(df[residual], label = 'Residuals')
  ax4.legend(loc = 'best')
  ax4.tick_params(axis = 'x', rotation = 45)
  plt.tight_layout()

  # Show graph
  plt.suptitle('Trend, Seasonal, and Residual Decomposition of %s' %(ts), 
               x = 0.5, 
               y = 1.05, 
               fontsize = 18)
  plt.show()
  plt.close()
  
  return
```


```python
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(df_processed['ts'], freq = 365)

df_processed.loc[:,'trend'] = decomposition.trend
df_processed.loc[:,'seasonal'] = decomposition.seasonal
df_processed.loc[:,'residual'] = decomposition.resid

plot_decomposition(df_processed, 
                   ts = 'ts', 
                   trend = 'trend',
                   seasonal = 'seasonal', 
                   residual = 'residual')

display(HTML("<b>The stationarity test of residual:</b>"))
test_stationarity(df_processed.dropna()['residual'])
```


![png](Time-series%20Preprocessing_files/Time-series%20Preprocessing_46_0.png)



<b>The stationarity test of residual:</b>



![png](Time-series%20Preprocessing_files/Time-series%20Preprocessing_46_2.png)


    Results of Dickey-Fuller Test:
    The time series is stationary at the p = 0.05 level.
    Test Statistic                   -5.064331
    p-value                           0.000017
    # Lags Used                      22.000000
    Number of Observations Used    2757.000000
    Critical Value (1%)              -3.432724
    Critical Value (5%)              -2.862589
    Critical Value (10%)             -2.567328
    dtype: float64
    

# After Preprocessing

Additional data considerations before choosing a model
*   Whether or not to incorporate external data
*   Whether or not to keep as univariate or multivariate (i.e., which features and number of features)
*   Outlier detection and removal
*   Missing value imputation

## Statistical models
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

  ### ARIMA models.
You may use ARIMA models when we know there is dependence between values and leverage that information to forecast.

**ARIMA = Auto-Regressive Integrated Moving Average**.   
Assumptions: The time-series is stationary.  
Depends on:
 1. Number of AR (Auto-Regressive) terms (p).
 2. Number of I (Integrated or Difference) terms (d).
 3. Number of MA (Moving Average) terms (q). 
  
 ### Facebook Prophet package.
[Facebook Prophet](https://facebook.github.io/prophet/), a tool that allows folks to forecast using additive or component models relatively easily.  It can also include things like:
* Day of week effects
* Day of year effects
* Holiday effects
* Trend trajectory
* Can do MCMC sampling

## Machine Learning.
*   **Ignore the time-series aspect completely and model using traditional machine learning modeling toolbox.** 
  *   *Examples.* Support Vector Machines (SVMs), Random Forest Regression, Gradient-Boosted Decision Trees (GBDTs).
*   **Hidden markov models (HMMs).**
*   **Other sequence-based models.**
*   **Gaussian processes (GPs).**
*   **Recurrent neural networks (RNNs).**

### Sklearn scalers
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
1. Applying statistical modeling and machine learning to perform time-series forecasting (https://goo.gl/r7CFcN). by Tamara Louie in PyData LA  October 2018 

2. The Simple Yet Practical Data Cleaning Codes - To solve the common scenarios of messy data (https://towardsdatascience.com/the-simple-yet-practical-data-cleaning-codes-ad27c4ce0a38) by Admond Lee, Jan 11 2019

