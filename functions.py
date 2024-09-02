
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import seaborn as sns

from definitions import quality_codes, quality_codes_inv, completeness_codes

def df_to_datetime(df):
    df['dateTime'] = pd.to_datetime(df['dateTime'])
    df['date'] = pd.to_datetime(df['date'])
    return df

def load_data(path):
    df = pd.read_csv(path)
    df = df_to_datetime(df)
    df = df.set_index('dateTime') # not fully tested
    df['quality'] = df['quality'].map(quality_codes)
    df['completeness'] = df['completeness'].map(completeness_codes)
    return df

def get_flow_data(location):
    cwd = os.getcwd()
    path = cwd + (f'\\data\\flow\\{location}-Upstream-flow-15min-Qualified.csv')
    df = load_data(path)
    return df

def get_rainfall_data_single(location):
    '''
    get rainfall data for a named rain gauge
    '''
    cwd = os.getcwd()
    path = cwd + (f'\\data\\rainfall\\{location}-rainfall-15min-Qualified.csv')
    df = load_data(path)
    return df

def get_rainfall_data(locations):
    '''
    get rainfall data for each of a set of locations
    '''
    rainfall_data = {}
    for location in locations:
        df = get_rainfall_data_single(location)
        rainfall_data[location] = df
    return rainfall_data

def resample_flow_hr(df):
    '''
    resample flow df by hour
    '''
    # mean of flow over preceeding hour (m^3/s)
    # issue - what to do with quality column
    df_hr = df.resample('h',
                                closed='right',
                                label='right'
                                ).agg({
                                    'value': 'mean',
                                    'quality': ['max', 'min']
                                })
    df_hr.columns = ['value', 'quality_max', 'quality_min']
    return df_hr

def resample_rf_hr(rf_dict):
    ''' 
    resample dict of rainfall dfs by hour 
    '''
    # sum over preceeding hour (mm)
    rf_hr = {}
    for location in rf_dict.keys():
        gauge_hr = rf_dict[location].resample('h',
                                                    closed='right',
                                                    label='right'
                                                    ).agg({
                                                        'value': 'sum',
                                                        'completeness': 'min',
                                                        'quality': ['max', 'min']
                                                    })
        gauge_hr.columns = ['value', 'completeness', 'quality_max', 'quality_min']
        rf_hr[location] = gauge_hr
    return rf_hr

def quality_count(data, year=None):
    '''
    count data points by quality label    
    '''
    if year:
        data = data[data.index.year == year]
    value_counts = data.quality.value_counts()
    value_counts.index = value_counts.index.map(quality_codes_inv)
    print(value_counts)

def plot_sample(df, sample_interval, minmax=False, logscale=False):
    df['min'] = df['value']
    df['max'] = df['value']
    df_sample = df.resample(sample_interval).agg({'value': 'mean',
                                                'max': 'max',
                                                'min': 'min'})

    plt.plot(df_sample.index, df_sample.value)
    plt.xlabel('year')
    plt.ylabel('average flow (m$^3$/s)')
    #plt.ylim(0,max(df_sample['value']))
    if minmax:
        plt.fill_between(df_sample.index, 
                 df_sample['min'], 
                 df_sample['max'], 
                 alpha=0.3)
        if logscale:
            plt.yscale('log')
    plt.gca().spines[:].set_visible(False)

def plot_histogram(df, quantile=None, bins=500):
    '''
    plots histogram of data up to specified quantile
    '''
    quantile = df['value'].quantile(quantile)

    # plot histogram of flow
    ax = df['value'].hist(bins=bins)
    scale = 1/1000
    ticks = ticker.FuncFormatter(lambda y, pos: '{0:g}'.format(y*scale))
    ax.yaxis.set_major_formatter(ticks)
    ax.set_xlim(left=0, right=quantile)
    ax.set_xlabel('Flow')
    ax.set_ylabel('Frequency (thousands)')
    ax.spines[:].set_visible(False)
    ax.grid(False)

def plot_quality(df, title=None, completeness=False, year=None):

    '''
    plots histogram of data quality (or completeness) by year for a dataframe of values and dates

    '''

    if year:
        df = df[df.index.year == year]
        period_name = 'Week'
        start = 1
        end = 54
        
    else:
        period_name = 'Year'
        # find start and end years
        start = df.date.min().year
        end = df.date.max().year + 1

    # define quality categories
    
    if completeness: 
        codes_dict = completeness_codes
    else:
        codes_dict = quality_codes
    
    categories = [category for category in codes_dict.keys()]


    df_quality = pd.DataFrame(columns = [period_name] + categories + ['total'])

    for period in range (start, end):
        if year:
            df_period = df[df.index.isocalendar().week == period]
        else:
            df_period = df[df.index.year == period]

        if completeness:
            value_counts = df_period['completeness'].value_counts()
        else:
            value_counts = df_period['quality'].value_counts()

        period_count = 0
        new_row = [period]
        for category in categories:
            count = value_counts.get(codes_dict[category], 0)
            period_count += count
            new_row.append(count)
        new_row.append(period_count) # add total count of data in year/week
        df_quality.loc[len(df_quality)] = new_row
        
    #%%
    # create a stacked bar chart showing data quality by year/week

    palette = ['#73e468', '#68bde4', 'yellow', 'orange', 'red']
    fig, ax = plt.subplots(figsize=(15, 5))
    labels = df_quality[period_name]
    ax.set_xlim(start-1, end)
    cum_quality_count = pd.Series([0] * len(df_quality))
    for i, (category, color) in enumerate(zip(categories, palette)):
        if df_quality[category].sum() > 0:
            widths = df_quality[category]
            bars = ax.bar(labels, widths, bottom=cum_quality_count, color=color, label=category)
            cum_quality_count += widths

    ax.legend(bbox_to_anchor=(1.01,0.9), frameon=False)
    ax.spines[:].set_visible(False)
    ax.set_xlabel(period_name)
    ax.set_ylabel('Data points')
    if title:
        plt.title(title)


def plot_seasonal(df, interval='daily', log=False, cmap=None, linewidth=None, linewidth_mean=5, title=None):
    '''
    for a dataframe of records, plot average values by day/week for each year
    
    parameters:
        df_flow - dataframe of records (flow or rainfall)
        interval - interval to average over - 'daily' or 'weekly'
        log - log-transform values (bool)
        cmap - colour palette for years (if not specified then uniform blue)
        linewidth - defaults are 0.2 for daily and 0.5 for weekly
        title - plot title (optional)

    '''


    ### Version 1
    '''
    df_1 = df[['date', 'value']]
    df_agg = df_1.groupby('date')['value'].mean().reset_index()
    df_agg['year'] = df_agg['date'].dt.year

    if interval == 'weekly':
        # aggregate data by week using year and week number
        df_agg['week'] = df_agg['date'].dt.isocalendar().week

        df_agg['group'] = df_agg['year'].astype(str) + df_agg['week'].astype(str)
        df_agg = df_agg.groupby('group').agg({
            'date': 'first',    # first date in each week
            'value': 'mean',    # mean of the value column
            'week': 'first',    # week number
            'year': 'first'     # year
        }).reset_index(drop=True)
        index = 'week'
    
    else:
        df_agg['day'] = df_agg['date'].dt.strftime('%m-%d')
        index = 'day'
    '''

    ### Version 2
    
    # aggregate data by day
    df = df[['date', 'value']]

    if interval == 'daily':
        df_agg = df.resample('D').mean(numeric_only=True)
        df_agg['day'] = df_agg.index.strftime('%m-%d')
        index = 'day'

    elif interval == 'weekly':
        df_agg = df.resample('W').mean(numeric_only=True)
        df_agg['week'] = df_agg.index.to_period('W').strftime('%W')
        index = 'week'
    
    elif interval == 'monthly':
        df_agg = df.resample('ME').mean(numeric_only=True)
        df_agg['month'] = df_agg.index.to_period('M').strftime('%m')
        index = 'month'

    else:
        raise ValueError('Interval given is invalid')
    '''
    # take log of values if required
    if log == True:
        df_agg['value'] = np.log(df_agg['value'])
    '''
    df_agg['year'] = df_agg.index.year


    # find max flow or log(flow) value
    y_max = df_agg['value'].max()

    # create dataframe of year vs day/week
    pivot_table = df_agg.pivot(index=index, columns='year', values='value')
    
    # generate colours for plot
    if cmap:
        cmap_ = plt.get_cmap(cmap)
        colours = cmap_(np.linspace(0, 1, len(pivot_table.columns)))
    else:
        colours = [sns.color_palette()[0]] * len(pivot_table.columns)
    fig, ax = plt.subplots(figsize=(10, 6))

    # Convert 'day' strings back to datetime objects for plotting
    if interval == 'daily':
        pivot_table.index = pd.to_datetime('2000-' + pivot_table.index)  # Dummy year 2000 for consistent formatting
    
    else:
        pivot_table.index = pivot_table.index.astype(int)
    
    # Set x-axis limits based on data
    min_date = pivot_table.index.min()
    max_date = pivot_table.index.max()
    ax.set_xlim(min_date, max_date)
    '''
    if log == True:
        ax.set_ylim(1, y_max*1.1)
    '''
    if log == False:
        ax.set_ylim(0, y_max*1.1)

    if not linewidth:
        if interval == 'daily':
            linewidth = 0.2
        else:
            linewidth = 0.5

    for i, column in enumerate(pivot_table.columns):
        ax.plot(pivot_table.index, 
                pivot_table[column], 
                label=column, 
                linewidth=linewidth, 
                color=colours[i]
                )
    ax.plot(pivot_table.index,
         pivot_table.mean(axis=1), 
         linewidth=linewidth*linewidth_mean, 
         color=sns.color_palette()[3])
    
    if interval == 'daily':
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        for tick in ax.xaxis.get_majorticklabels():
            tick.set_horizontalalignment("left")
        ax.set_xlabel('Date')
    elif interval == 'weekly':
        ax.set_xlabel('Week number')
    else:
        ax.set_xlabel('Month number')

    if log == True:
        ax.set_yscale('log')
        ax.set_ylabel(f'Mean {interval} flow (log scale)')
    else:
        ax.set_ylabel(f'Mean {interval} flow')
    ax.spines[:].set_visible(False)

    if title:
        plt.title(title)
    #ax.legend()

    # Show the plot
    plt.show()


def rainfall_sample(rainfall_data, loc='all', sample_size=1000):

    '''
    takes an input set of rainfall dataframes and returns a sample of data of given size
    '''
    
    quality_cols = ['completeness', 'quality_min', 'quality_max']

    # take a sample of (sample_size) rows from initial location
    if loc == 'all':
        loc_init = 'Bury'
    else:
        loc_init = loc
    df_gauge = rainfall_data[loc_init]
    sample_inds = np.random.permutation(len(df_gauge))[:sample_size]
    rf_sample = df_gauge.iloc[sample_inds]

    # take the average of each rain gauge in catchment for the sample of rows
    if loc == 'all':
        rf_sample = rf_sample.rename(columns={'value':'rf_Bury'})
        new_col_names = ['rf_Bury']
        for location in rainfall_data.keys():
            if location != 'Bury':
                location_df = rainfall_data[location]
                col_name = f'rf_{location}'
                new_col_names.append(col_name)
                rf_sample = pd.merge(rf_sample, 
                                    location_df[['value']].rename(columns={'value':col_name}),
                                    how='left',
                                    on='dateTime'
                                    )       
            
        rf_sample['rf_mean'] = rf_sample[new_col_names].mean(axis=1)
        rf_sample = rf_sample[['rf_mean'] + new_col_names]
    else:
        rf_sample = rf_sample[['value'] + quality_cols]
    return rf_sample

def offset_flow(rf_sample, df_flow, offsets):

    '''
    for a dataframe of (sampled) rainfall data, adds columns with flow data offset by given numbers of hours
    '''

    rf_sample = rf_sample.copy() # avoids warning message
    for offset in offsets:
        timedelta_val = pd.Timedelta(hours = offset)
        rf_sample.loc[:, f'dt_{offset}h'] = rf_sample.index + timedelta_val

        rf_sample = pd.merge(rf_sample, 
                            df_flow[['value']].rename(columns={'value': f'flow_{offset}h'}), 
                            left_on=f'dt_{offset}h',
                            right_index=True,
                            how='left')
    return rf_sample

