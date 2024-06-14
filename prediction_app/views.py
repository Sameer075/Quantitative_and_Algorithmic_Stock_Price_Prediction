from django.shortcuts import render ,redirect
from django.contrib.auth import logout
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse, JsonResponse
from .models import user
from django.db.models import Q
from .forms import *
import json
import pandas as pd
import numpy as np
import pandas_ta as ta
from django.views.decorators.csrf import csrf_exempt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import datetime
from tvDatafeed import TvDatafeed, Interval
from random import randint
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix

from django.core.mail import send_mail
from allauth.socialaccount.models import SocialAccount

tv = TvDatafeed()

# Define symbol mapping
symbol_mapping = {
    "HDFC Bank Limited": "HDFCBANK",
    "Infosys Limited": "INFY",
    "ITC Limited": "ITC",
    "State Bank of India": "SBIN",
    "Tata Consultancy Services Limited": "TCS",
    "Wipro Limited": "WIPRO",
    "BHARTIARTL": "BHARTIARTL",
    "BANKNIFTY1!":"BANKNIFTY1!",
}

holidays = [
    '2023-01-26',  # Republic Day
    '2023-03-07',  # Holi
    '2023-08-15',  # Independence Day
    '2023-10-02',  # Gandhi Jayanti
    '2023-11-12',  # Diwali
    '2023-12-25',  # Christmas
    # Add more holidays as required
]

# Create your views here.
def login(request):
    if request.user.is_authenticated:
        return redirect("/home")
    return render(request,'login.html')

def logcode(request):
    name=request.GET['user_name']
    pwd=request.GET['user_pwd']
    if user.objects.filter(name=name,pwd=pwd):
       xx=user.objects.get(name=name)
       if request.method == 'POST':
        message = request.POST.get('message')
        emailID = request.POST.get('emailID')

        if message and emailID:
            send_mail(
                emailID,
                message,
                'ytcodecreation@gmail.com',
                ['ytcodecreation@gmail.com'],
                fail_silently=False,
            )

       return render(request,'home.html')
    return render(request,'login.html')

def demo(request):
    return HttpResponse('testing 1st page')

def logout_view(request):
    logout(request)
    return redirect('/')

@login_required
def home(request):
    
    extra_data = SocialAccount.objects.get(user=request.user).extra_data
    picture_url =extra_data['picture']
    if request.method == 'POST':
        message = request.POST.get('message')
        emailID = request.POST.get('emailID')

        if  message and emailID:
            send_mail(
                emailID,
                message,
                emailID,
                ['ytcodecreation@gmail.com'],
                fail_silently=False,
            )

    return render(request,'home.html',{'picture_url': picture_url})


def fetch_data(symbol2, interval=Interval.in_daily, n_bars=5000):
    data = tv.get_hist(symbol=symbol2, exchange='NSE', interval=interval, n_bars=n_bars, extended_session=False)
    return data

# Function to check if a given date is a holiday
def is_holiday(date):
    return date.strftime('%Y-%m-%d') in holidays

# Function to get the last trading day's intraday session
def get_last_trading_day_data(current_time):
    previous_day = current_time - pd.Timedelta(days=1)
    
    while previous_day.dayofweek > 4 or is_holiday(previous_day):
        previous_day -= pd.Timedelta(days=1)
        
    start_time = previous_day.normalize() + pd.Timedelta(hours=9, minutes=15)
    end_time = previous_day.normalize() + pd.Timedelta(hours=15, minutes=30)
    
    return start_time, end_time


def fetch_data2(symbol3, interval=Interval.in_1_minute):
    # Get the current time
    current_time = pd.Timestamp.now()

    # Determine if it's after market close (3:30 PM)
    market_close_time = current_time.normalize() + pd.Timedelta(hours=15, minutes=30)

    # Determine the start and end times for the intraday session
    if current_time > market_close_time and current_time.dayofweek < 5 and not is_holiday(current_time):
        # Fetch today's data (full intraday session)
        start_time = current_time.normalize() + pd.Timedelta(hours=9, minutes=15)
        end_time = market_close_time
    else:
        # Fetch the last trading day's data (full intraday session)
        start_time, end_time = get_last_trading_day_data(current_time)

    # Fetch sufficient historical data to cover the needed range
    n_bars = 780  # Enough bars to cover the session
    data = tv.get_hist(symbol=symbol3, exchange="NSE", interval=interval, n_bars=n_bars, extended_session=False)
    # Filter the data to get only the desired session
    data = data.loc[start_time:end_time]
    
    return data

def fetch_data3(symbol4, interval=Interval.in_5_minute, n_bars=800):
    data = tv.get_hist(symbol=symbol4, exchange='NSE', interval=interval, n_bars=n_bars, extended_session=False)
    return data

@login_required
def linear_regression_model(request):
    # Load stock symbols from database or define them here
    stock_symbols = ["HDFC Bank Limited", "Infosys Limited", "ITC Limited", "State Bank of India", "Tata Consultancy Services Limited", "Wipro Limited"]
    return render(request, 'linear_regression_model.html', {'stockSymbols': stock_symbols})

@login_required
def random_forest_model(request):
    # Load stock symbols from database or define them here
    stock_symbols = ["HDFC Bank Limited", "Infosys Limited", "ITC Limited", "State Bank of India", "Tata Consultancy Services Limited", "Wipro Limited"]
    return render(request, 'random_forest_model.html', {'stockSymbols': stock_symbols})

@login_required
def ensemble_model(request):
    # Load stock symbols from database or define them here
    stock_symbols = ["HDFC Bank Limited", "Infosys Limited", "ITC Limited", "State Bank of India", "Tata Consultancy Services Limited", "Wipro Limited"]
    return render(request, 'ensemble_model.html', {'stockSymbols': stock_symbols})

@login_required
def markovian_model(request):
    # Load stock symbols from database or define them here
    stock_symbols = ["HDFC Bank Limited", "Infosys Limited", "ITC Limited", "State Bank of India", "Tata Consultancy Services Limited", "Wipro Limited"]
    return render(request, 'markovian_model.html', {'stockSymbols': stock_symbols})

@login_required
def intraday_strategy(request):
    # Load stock symbols from database or define them here
    stock_symbols = ["HDFC Bank Limited", "Infosys Limited", "ITC Limited", "State Bank of India", "Tata Consultancy Services Limited", "Wipro Limited", "BHARTIARTL", "BANKNIFTY1!"]
    return render(request, 'intraday_strategy.html', {'stockSymbols': stock_symbols})

def quantative_intraday_strategy(request):
    # Load stock symbols from database or define them here
    stock_symbols = ["HDFC Bank Limited", "Infosys Limited", "ITC Limited", "State Bank of India", "Tata Consultancy Services Limited", "Wipro Limited", "BHARTIARTL"]
    return render(request, 'quantative_intraday_strategy.html', {'stockSymbols': stock_symbols})

@login_required
def community(request):
    extra_data = SocialAccount.objects.get(user=request.user).extra_data
    picture_url =extra_data['picture']
    print("Hello community")
    return render(request,'community.html',{'picture_url': picture_url})

# Function to create and fit scaler
def create_and_fit_scaler(df):
    scaler = StandardScaler()
    scaler.fit(df)
    return scaler

def preprocess_and_predict(data):
    df = pd.DataFrame(data)
    # df['datetime'] = pd.to_datetime(df['datetime'])

    # Reset index to get 'datetime' as a column
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'datetime'}, inplace=True)
    # Handle missing values in the DataFrame
    df.dropna(inplace=True)  # Drop rows with missing values

    # Filter the dataset to include only data from the last 10 years
    end_date = df['datetime'].max()
    start_date = end_date - pd.DateOffset(years=10)
    df = df[df['datetime'] >= start_date]


    # Calculating Volume Profile
    poc_price = df.loc[df['volume'].idxmax(), 'close']
    z = df['close'].to_numpy()
    x = abs(z - poc_price)**2
    std = np.sqrt(np.mean(x))
    poc_threshold = std / 2
    lower_threshold = poc_price - poc_threshold
    upper_threshold = poc_price + poc_threshold
    df['Position_VP'] = np.where(df['close'] < lower_threshold, -1,
                                 np.where(df['close'] > upper_threshold, 1, 0))

    # Calculate other technical indicators
    df['SMA_50'] = ta.sma(df['close'], length=50)
    df['SMA_200'] = ta.sma(df['close'], length=200)
    df['Position_SMA'] = np.where(df['SMA_50'] > df['SMA_200'], 1, -1)
    df['RSI'] = ta.rsi(df['close'], length=14)
    df['Position_RSI'] = np.where((df['RSI'] < 20), 2,
                                  np.where((df['RSI'] >= 21) & (df['RSI'] < 30), 1,
                                           np.where((df['RSI'] > 80), -2,
                                                    np.where((df['RSI'] >= 70) & (df['RSI'] <= 79), -1, 0))))

    # Apply scaling to features
    features = df[['volume', 'Position_VP', 'SMA_200', 'Position_SMA', 'RSI', 'Position_RSI']].fillna(0).values

    # Create and fit scaler
    scaler = create_and_fit_scaler(features)
    features_scaled = scaler.transform(features)  # Scale features using the scaler

    # Splitting the dataset into 80% training and 20% testing
    x_train, x_test, y_train, y_test = train_test_split(features_scaled, df['close'].values, test_size=0.2, random_state=101)

    # Convert x_test to DataFrame
    x_test_df = pd.DataFrame(x_test, columns=['volume', 'Position_VP', 'SMA_200', 'Position_SMA', 'RSI', 'Position_RSI'])

    # Create forecast dates for the test set
    forecast_dates = df.iloc[x_test_df.index]['datetime'].tolist()

    # Train a linear regression model
    lrmodel = LinearRegression()
    lrmodel.fit(x_train, y_train)

    # Make predictions on the test set
    predictions = lrmodel.predict(x_test)

    # Evaluate the model
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    
    return {'predicted_close': predictions.tolist(), 'actual_close': y_test.tolist(), 'forecast_dates': forecast_dates,
            'metrics': {'Mean Absolute Error': mae, 'Mean Squared Error': mse,
                        'Root Mean Squared Error': rmse, 'R-squared': r2}}

    
def preprocess_and_predict_random_forest(data):
    df = pd.DataFrame(data)
    #df['Date'] = pd.to_datetime(df['Date'])

    # Reset index to get 'datetime' as a column
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'datetime'}, inplace=True)
    # Handle missing values in the DataFrame
    df.dropna(inplace=True)  # Drop rows with missing values

    # Filter the dataset to include only data from the last 10 years
    end_date = df['datetime'].max()
    start_date = end_date - pd.DateOffset(years=10)
    df = df[df['datetime'] >= start_date]


    # Calculating Volume Profile
    poc_price = df.loc[df['volume'].idxmax(), 'close']
    z = df['close'].to_numpy()
    x = abs(z - poc_price)**2
    std = np.sqrt(np.mean(x))
    poc_threshold = std / 2
    lower_threshold = poc_price - poc_threshold
    upper_threshold = poc_price + poc_threshold
    df['Position_VP'] = np.where(df['close'] < lower_threshold, -1,
                                 np.where(df['close'] > upper_threshold, 1, 0))

    # Calculate other technical indicators
    df['SMA_50'] = ta.sma(df['close'], length=50)
    df['SMA_200'] = ta.sma(df['close'], length=200)
    df['Position_SMA'] = np.where(df['SMA_50'] > df['SMA_200'], 1, -1)
    df['RSI'] = ta.rsi(df['close'], length=14)
    df['Position_RSI'] = np.where((df['RSI'] < 20), 2,
                                  np.where((df['RSI'] >= 21) & (df['RSI'] < 30), 1,
                                           np.where((df['RSI'] > 80), -2,
                                                    np.where((df['RSI'] >= 70) & (df['RSI'] <= 79), -1, 0))))

    # Apply scaling to features
    features = df[['volume', 'Position_VP', 'SMA_200', 'Position_SMA', 'RSI', 'Position_RSI']].fillna(0).values

    # Create and fit scaler
    scaler = create_and_fit_scaler(features)
    features_scaled = scaler.transform(features)  # Scale features using the scaler

    # Splitting the dataset into 80% training and 20% testing
    x_train, x_test, y_train, y_test = train_test_split(features_scaled, df['close'].values, test_size=0.2, random_state=101)

    # Convert x_test to DataFrame
    x_test_df = pd.DataFrame(x_test, columns=['volume', 'Position_VP', 'SMA_200', 'Position_SMA', 'RSI', 'Position_RSI'])

    # Create forecast dates for the test set
    forecast_dates = df.iloc[x_test_df.index]['datetime'].tolist()

    # Create a Random Forest Regression Model
    rfmodel = RandomForestRegressor(random_state=42)
    rfmodel.fit(x_train, y_train)

    # Make predictions on the train set
    predictions = rfmodel.predict(x_test)

    # Evaluate the model
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    
    return {'predicted_close': predictions.tolist(), 'actual_close': y_test.tolist(), 'forecast_dates': forecast_dates,
            'metrics': {'Mean Absolute Error': mae, 'Mean Squared Error': mse,
                        'Root Mean Squared Error': rmse, 'R-squared': r2}}

# Base models for stacking ensemble
gbr_model = GradientBoostingRegressor()
xgb_model = XGBRegressor(verbosity=0)
lgbm_model = LGBMRegressor(verbose=-1)
catboost_model = CatBoostRegressor(verbose=False)

base_models = [('GradientBoosting', gbr_model), 
               ('XGBoost', xgb_model), 
               ('LightGBM', lgbm_model),
               ('CatBoost', catboost_model)]
meta_model = LinearRegression()

def create_and_fit_scaler(features):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(features)
    return scaler

def preprocess_and_predict_ensemble(data):
    df = pd.DataFrame(data)
    #df['Date'] = pd.to_datetime(df['Date'])

    # Reset index to get 'datetime' as a column
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'datetime'}, inplace=True)
    # Handle missing values in the DataFrame
    df.dropna(inplace=True)  # Drop rows with missing values

    # Filter the dataset to include only data from the last 10 years
    end_date = df['datetime'].max()
    start_date = end_date - pd.DateOffset(years=10)
    df = df[df['datetime'] >= start_date]

    # Calculating Volume Profile
    poc_price = df.loc[df['volume'].idxmax(), 'close']
    z = df['close'].to_numpy()
    x = abs(z - poc_price) ** 2
    std = np.sqrt(np.mean(x))
    poc_threshold = std / 2
    lower_threshold = poc_price - poc_threshold
    upper_threshold = poc_price + poc_threshold
    df['Position_VP'] = np.where(df['close'] < lower_threshold, -1,
                                 np.where(df['close'] > upper_threshold, 1, 0))

    # Calculate other technical indicators
    df['SMA_50'] = ta.sma(df['close'], length=50)
    df['SMA_200'] = ta.sma(df['close'], length=200)
    df['Position_SMA'] = np.where(df['SMA_50'] > df['SMA_200'], 1, -1)
    df['RSI'] = ta.rsi(df['close'], length=14)
    df['Position_RSI'] = np.where((df['RSI'] < 20), 2,
                                  np.where((df['RSI'] >= 21) & (df['RSI'] < 30), 1,
                                           np.where((df['RSI'] > 80), -2,
                                                    np.where((df['RSI'] >= 70) & (df['RSI'] <= 79), -1, 0))))

    # Apply scaling to features
    features = df[['volume', 'Position_VP', 'SMA_200', 'Position_SMA', 'RSI', 'Position_RSI']].fillna(0).values

    # Create and fit scaler
    scaler = create_and_fit_scaler(features)
    features_scaled = scaler.transform(features)  # Scale features using the scaler

    # Splitting the dataset into 80% training and 20% testing
    x_train, x_test, y_train, y_test = train_test_split(features_scaled, df['close'].values, test_size=0.2, random_state=101)

    # Convert x_test to DataFrame
    x_test_df = pd.DataFrame(x_test, columns=['volume', 'Position_VP', 'SMA_200', 'Position_SMA', 'RSI', 'Position_RSI'])

    # Create forecast dates for the test set
    forecast_dates = df.iloc[x_test_df.index]['datetime'].tolist()

    # Training the ensemble
    predictions_train = []
    for model_name, model in base_models:
        model.fit(x_train, y_train)
        train_pred = model.predict(x_train)
        predictions_train.append(train_pred)

    # Stack predictions for training meta-model
    stacked_predictions_train = np.column_stack(predictions_train)
    meta_model.fit(stacked_predictions_train, y_train)

    # Validation and Testing
    # Make predictions on the test set
    predictions_test = []
    for model_name, model in base_models:
        test_pred = model.predict(x_test)
        predictions_test.append(test_pred)

    # Stack predictions for testing meta-model
    stacked_predictions_test = np.column_stack(predictions_test)

    # Make final predictions using the meta-model
    ensemble_predictions = meta_model.predict(stacked_predictions_test)

    # Evaluate the model
    mae = mean_absolute_error(y_test, ensemble_predictions)
    mse = mean_squared_error(y_test, ensemble_predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, ensemble_predictions)
    
    return {'predicted_close': ensemble_predictions.tolist(), 'actual_close': y_test.tolist(), 'forecast_dates': forecast_dates,
            'metrics': {'Mean Absolute Error': mae, 'Mean Squared Error': mse,
                        'Root Mean Squared Error': rmse, 'R-squared': r2}}

def preprocess_and_predict_markovian(data):
    gspc_df = pd.DataFrame(data)

    # Reset index to get 'datetime' as a column
    gspc_df.reset_index(inplace=True)
    gspc_df.rename(columns={'index': 'datetime'}, inplace=True)
    # Handle missing values in the DataFrame
    gspc_df.dropna(inplace=True)  # Drop rows with missing values

    # Filter the dataset to include only data from the last 10 years
    end_date = gspc_df['datetime'].max()
    start_date = end_date - pd.DateOffset(years=10)
    gspc_df = gspc_df[gspc_df['datetime'] >= start_date]

    # take random sets of sequential rows 
    new_set = []
    for row_set in range(0, 10000):
        if row_set%2000==0: print(row_set)
        row_quant = randint(10, 30)
        row_start = randint(0, len(gspc_df)-row_quant)
        market_subset = gspc_df.iloc[row_start:row_start+row_quant]

        Close_Date = max(market_subset['datetime'])
        if row_set%2000==0: print(Close_Date)
    
        # Close_Gap = (market_subset['Close'] - market_subset['Close'].shift(1)) / market_subset['Close'].shift(1)
        Close_Gap = market_subset['close'].pct_change()
        High_Gap = market_subset['high'].pct_change()
        Low_Gap = market_subset['low'].pct_change() 
        Volume_Gap = market_subset['volume'].pct_change() 
        Daily_Change = (market_subset['close'] - market_subset['open']) / market_subset['open']
        Outcome_Next_Day_Direction = (market_subset['volume'].shift(-1) - market_subset['volume'])
    
        new_set.append(pd.DataFrame({'Sequence_ID':[row_set]*len(market_subset),
                            'Close_Date':[Close_Date]*len(market_subset),
                           'Close_Gap':Close_Gap,
                           'High_Gap':High_Gap,
                           'Low_Gap':Low_Gap,
                           'Volume_Gap':Volume_Gap,
                           'Daily_Change':Daily_Change,
                           'Outcome_Next_Day_Direction':Outcome_Next_Day_Direction}))
    
    new_set_df = pd.concat(new_set)
    new_set_df = new_set_df.dropna(how='any') 
    
    # create sequences
    # simplify the data by binning values into three groups
 
    # Close_Gap
    new_set_df['Close_Gap_LMH'] = pd.qcut(new_set_df['Close_Gap'], 3, labels=["L", "M", "H"])

    # High_Gap - not used in this example
    new_set_df['High_Gap_LMH'] = pd.qcut(new_set_df['High_Gap'], 3, labels=["L", "M", "H"])

    # Low_Gap - not used in this example
    new_set_df['Low_Gap_LMH'] = pd.qcut(new_set_df['Low_Gap'], 3, labels=["L", "M", "H"])

    # Volume_Gap
    new_set_df['Volume_Gap_LMH'] = pd.qcut(new_set_df['Volume_Gap'], 3, labels=["L", "M", "H"])
 
    # Daily_Change
    new_set_df['Daily_Change_LMH'] = pd.qcut(new_set_df['Daily_Change'], 3, labels=["L", "M", "H"])

    # new set
    new_set_df = new_set_df[["Sequence_ID", 
                         "Close_Date", 
                         "Close_Gap_LMH", 
                         "Volume_Gap_LMH", 
                         "Daily_Change_LMH", 
                         "Outcome_Next_Day_Direction"]]

    new_set_df['Event_Pattern'] = new_set_df['Close_Gap_LMH'].astype(str) + new_set_df['Volume_Gap_LMH'].astype(str) + new_set_df['Daily_Change_LMH'].astype(str)
    
    compressed_set = new_set_df.groupby(['Sequence_ID', 
                                     'Close_Date'])['Event_Pattern'].apply(lambda x: "{%s}" % ', '.join(x)).reset_index()

    compressed_outcomes = new_set_df.groupby(['Sequence_ID', 'Close_Date'])['Outcome_Next_Day_Direction'].mean()
    compressed_outcomes = compressed_outcomes.to_frame().reset_index()

    compressed_set = pd.merge(compressed_set, compressed_outcomes, on= ['Sequence_ID', 'Close_Date'], how='inner')

    compressed_set['Event_Pattern'] = [''.join(e.split()).replace('{','')
                                   .replace('}','') for e in compressed_set['Event_Pattern'].values]
    
    compressed_set_validation = compressed_set[compressed_set['Close_Date'] >= datetime.datetime.now() 
                                           - datetime.timedelta(days=90)] # Sys.Date()-90 

    compressed_set = compressed_set[compressed_set['Close_Date'] < datetime.datetime.now() 
                                           - datetime.timedelta(days=90)] 

    # drop date field
    compressed_set = compressed_set[['Sequence_ID', 'Event_Pattern','Outcome_Next_Day_Direction']]
    compressed_set_validation = compressed_set_validation[['Sequence_ID', 'Event_Pattern','Outcome_Next_Day_Direction']]

    print(len(compressed_set['Outcome_Next_Day_Direction']))
    len(compressed_set[abs(compressed_set['Outcome_Next_Day_Direction']) > 10000000])

    # compressed_set = compressed_set[abs(compressed_set['Outcome_Next_Day_Direction']) > 10000000]
    compressed_set['Outcome_Next_Day_Direction'] = np.where((compressed_set['Outcome_Next_Day_Direction'] > 0), 1, 0)
    compressed_set_validation['Outcome_Next_Day_Direction'] = np.where((compressed_set_validation['Outcome_Next_Day_Direction'] > 0), 1, 0)

    # create two data sets - won/not won
    compressed_set_pos = compressed_set[compressed_set['Outcome_Next_Day_Direction']==1][['Sequence_ID', 'Event_Pattern']]
    compressed_set_neg = compressed_set[compressed_set['Outcome_Next_Day_Direction']==0][['Sequence_ID', 'Event_Pattern']]

    flat_list = [item.split(',') for item in compressed_set['Event_Pattern'].values ]
    unique_patterns = ','.join(str(r) for v in flat_list for r in v)
    unique_patterns = list(set(unique_patterns.split(',')))


    
    def build_transition_grid(compressed_grid, unique_patterns):
        patterns = []
        counts = []

        for from_event in unique_patterns:
            for to_event in unique_patterns:
                pattern = f"{from_event},{to_event}"
                ids_matches = compressed_grid[compressed_grid['Event_Pattern'].str.contains(pattern)]
                found = ids_matches['Event_Pattern'].str.count(pattern).sum()
                patterns.append(pattern)
                counts.append(found)

        grid_df = pd.DataFrame({'pairs': patterns, 'counts': counts})

        # Split 'pairs' into 'x' and 'y' coordinates
        grid_df[['x', 'y']] = grid_df['pairs'].str.split(',', n=1, expand=True)

        # Pivot the DataFrame to create the grid
        grid_df = grid_df.pivot(index='x', columns='y', values='counts').fillna(0)

        # Normalize by row sums
        grid_df = grid_df.div(grid_df.sum(axis=1), axis=0)

        return grid_df

    grid_pos = build_transition_grid(compressed_set_pos, unique_patterns) 
    grid_neg = build_transition_grid(compressed_set_neg, unique_patterns) 
    
    def safe_log(x,y):
        try:
            lg = np.log(x/y)
        except:
            lg = 0
        return lg

    # predict on out of sample data
    actual = []
    predicted = []
    for seq_id in compressed_set_validation['Sequence_ID'].values:
        patterns = compressed_set_validation[compressed_set_validation['Sequence_ID'] == seq_id]['Event_Pattern'].values[0].split(',')
        pos = []
        neg = []
        log_odds = []
    
        for id in range(0, len(patterns)-1):
            # get log odds
            # logOdds = log(tp(i,j) / tn(i,j)
            if (patterns[id] in list(grid_pos) and patterns[id+1] in list(grid_pos) and patterns[id] in list(grid_neg) and patterns[id+1] in list(grid_neg)):
                
                numerator = grid_pos[patterns[id]][patterns[id+1]]
                denominator = grid_neg[patterns[id]][patterns[id+1]]
                if (numerator == 0 and denominator == 0):
                    log_value =0
                elif (denominator == 0):
                    log_value = np.log(numerator / 0.00001)
                elif (numerator == 0):
                    log_value = np.log(0.00001 / denominator)
                else:
                    log_value = np.log(numerator/denominator)
            else:
                log_value = 0
        
            log_odds.append(log_value)
        
            pos.append(numerator)
            neg.append(denominator)
      
    
        actual.append(compressed_set_validation[compressed_set_validation['Sequence_ID']==seq_id]['Outcome_Next_Day_Direction'].values[0])
        predicted.append(sum(log_odds))
    
    # Calculate confusion matrix
    confusion = confusion_matrix(actual, [1 if p > 0 else 0 for p in predicted])
    # Calculate metrics
    accuracy = accuracy_score(actual, [1 if p > 0 else 0 for p in predicted])
    precision = precision_score(actual, [1 if p > 0 else 0 for p in predicted])
    recall = recall_score(actual, [1 if p > 0 else 0 for p in predicted])
    f1 = f1_score(actual, [1 if p > 0 else 0 for p in predicted])
    
    # ROC Curve and AUC
    fpr, tpr, _ = roc_curve(actual, predicted)
    roc_auc = auc(fpr, tpr)
    
    # Convert lists to regular Python integers
    predicted_close = [int(x) for x in predicted]
    actual_close = [int(x) for x in actual]
    print(predicted_close)
    print(actual_close)

    # Prepare the data to send back to the client
    result_data = {
        'predicted_close': predicted_close,
        'actual_close': actual_close,
        'metrics': {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'confusionMatrix': confusion.tolist(),
            'roc': {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'auc': roc_auc
            }
            # Other metrics can also be included here
        }
    }

    return result_data


# def calculate_tpo_counts(data):
#     tpo_counts = {}
#     for index, row in data.iterrows():
#         high = row['high']
#         low = row['low']
#         for price in np.arange(low, high + 0.01, 0.01):  # Increment by 0.01 for precision
#             price = round(price, 2)  # Ensure consistency in rounding
#             if price not in tpo_counts:
#                 tpo_counts[price] = 0
#             tpo_counts[price] += 1
#     return tpo_counts

# def preprocess_and_calculate_levels(data):
#     data_30min = data.resample('30min').agg({'high': 'max', 'low': 'min', 'close': 'last'})
#     data_30min.dropna(inplace=True)
#     data_30min['TPO'] = [chr(65 + i % 26) for i in range(len(data_30min))]  # Cycle through A-Z for TPO

#     tpo_counts = calculate_tpo_counts(data_30min)

#     # Price of Control (POC)
#     poc = sum([p * c for p, c in tpo_counts.items()]) / sum(tpo_counts.values())
#     print(poc)

#     # High and Low price of Intraday Dataset (data)
#     high = data['high'].max()
#     low = data['low'].min()

#     # Positive Gaussian Curve
#     median = (high + low) / 2
#     diff = median - poc
#     diff2 = diff * 2
#     new_high = high + diff2

#     numbers_positive = np.array([new_high, median])
#     mean_positive = np.mean(numbers_positive)
#     poc_2 = mean_positive
#     squared_diff_positive = (numbers_positive - poc_2) ** 2
#     variance_positive = np.sum(squared_diff_positive) / len(numbers_positive)
#     std_positive = np.sqrt(variance_positive)
#     std_2 = std_positive / 2

#     a1 = poc_2 + std_2
#     b1 = poc_2 - std_2

#     # Negative Gaussian Curve
#     new_low = low - diff2

#     numbers_negative = np.array([median, new_low])
#     mean_negative = np.mean(numbers_negative)
#     poc_3 = mean_negative
#     squared_diff_negative = (numbers_negative - poc_3) ** 2
#     variance_negative = np.sum(squared_diff_negative) / len(numbers_negative)
#     std_negative = np.sqrt(variance_negative)
#     std_3 = std_negative / 2

#     a2 = poc_3 + std_3
#     b2 = poc_3 - std_3

#     # Construct levels_data dictionary
#     prices = sorted(tpo_counts.keys())
#     counts = [tpo_counts[price] for price in prices]

#     levels_data = {
#         'prices': prices,
#         'counts': counts,
#         'positive_levels': {
#             'Strong SELL (New High)': new_high,
#             'SELL (a1)': a1,
#             'POC (poc_2)': poc_2,
#             'b1': b1,
#             'Low (Median)': median
#         },
#         'negative_levels': {
#             'High (Median)': median,
#             'a2': a2,
#             'POC (poc_3)': poc_3,
#             'BUY (b2)': b2,
#             'Strong BUY (New Low)': new_low
#         }
#     }

#     return levels_data

def calculate_tpo_counts(data, tpo_profile):
    tpo_counts = {}
    for index, row in data.iterrows():
        high = row['high']
        low = row['low']
        for level, tpo_list in tpo_profile.items():
            if low <= level <= high:
                for tpo in tpo_list:
                    if level not in tpo_counts:
                        tpo_counts[level] = 0
                    tpo_counts[level] += 1
    return tpo_counts

def preprocess_and_calculate_levels(data):
    block_size = '30min'  # Can be 5min, 10min, 15min, 30min, 1H, 2H, 4H
    row_size = 'Auto'     # Number of ticks per row

    # Resample data to the specified block size
    resampled_data = data.resample(block_size).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    # Calculate row size automatically
    def calculate_row_size(df, ticks_per_point=1):
        min_tick = df['high'].max() - df['low'].min()
        min_tick_range = (df['high'].max() - df['low'].min()) / ticks_per_point
        row_ticks = min_tick_range / 80
        if 1 <= row_ticks <= 100:
            increment = 5
        elif 100 <= row_ticks <= 1000:
            increment = 50
        elif 1000 <= row_ticks <= 10000:
            increment = 500
        elif 10000 <= row_ticks <= 100000:
            increment = 5000
        else:
            increment = 50000
        ticks_per_row = max(round(row_ticks / increment) * increment, 1)  # Ensure ticks_per_row is at least 1
        return ticks_per_row

    if row_size == 'Auto':
        ticks_per_row = calculate_row_size(resampled_data)
    else:
        ticks_per_row = int(row_size)

    # Create price levels for TPO
    price_levels = np.arange(resampled_data['low'].min(), resampled_data['high'].max(), ticks_per_row)
    tpo_profile = {level: [] for level in price_levels}

    # Assign TPO letters
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    letter_index = 0

    for idx, row in resampled_data.iterrows():
        high = row['high']
        low = row['low']
        for level in price_levels:
            if low <= level <= high:
                tpo_profile[level].append(letters[letter_index])
        letter_index = (letter_index + 1) % len(letters)

    # Calculate total number of TPO blocks
    total_blocks = sum(len(tpo_profile[level]) for level in tpo_profile)

    # Calculate VA target
    value_area_percent = 70  # Default to 70%
    va_target = total_blocks * value_area_percent / 100

    # Find the POC (Point of Control)
    poc = max(tpo_profile, key=lambda level: len(tpo_profile[level]))
    print(poc)
    tpo_counts = calculate_tpo_counts(data, tpo_profile)

    # Calculate VAH and VAL
    sorted_levels = sorted(tpo_profile.keys())
    va_count = len(tpo_profile[poc])
    va_high = poc
    va_low = poc

    def count_tpo_blocks(level):
        return len(tpo_profile[level])

    while va_count < va_target:
        next_high = next((lvl for lvl in sorted_levels if lvl > va_high), None)
        next_low = next((lvl for lvl in sorted_levels if lvl < va_low), None)

        if next_high and next_low:
            if count_tpo_blocks(next_high) >= count_tpo_blocks(next_low):
                va_high = next_high
                va_count += count_tpo_blocks(next_high)
            else:
                va_low = next_low
                va_count += count_tpo_blocks(next_low)
        elif next_high:
            va_high = next_high
            va_count += count_tpo_blocks(next_high)
        elif next_low:
            va_low = next_low
            va_count += count_tpo_blocks(next_low)
        else:
            break

    # VAH and VAL
    vah = va_high
    print(vah)
    val = va_low
    print(val)

    # Positive Gaussian Curve
    high = resampled_data['high'].max()
    low = resampled_data['low'].min()
    median = (high + low) / 2
    diff = median - poc
    diff2 = diff * 2
    new_high = high + diff2

    numbers_positive = np.array([new_high, median])
    mean_positive = np.mean(numbers_positive)
    poc_2 = mean_positive
    squared_diff_positive = (numbers_positive - poc_2) ** 2
    variance_positive = np.sum(squared_diff_positive) / len(numbers_positive)
    std_positive = np.sqrt(variance_positive)
    std_2 = std_positive / 2

    a1 = poc_2 + std_2
    b1 = poc_2 - std_2

    # Negative Gaussian Curve
    new_low = low - diff2

    numbers_negative = np.array([median, new_low])
    mean_negative = np.mean(numbers_negative)
    poc_3 = mean_negative
    squared_diff_negative = (numbers_negative - poc_3) ** 2
    variance_negative = np.sum(squared_diff_negative) / len(numbers_negative)
    std_negative = np.sqrt(variance_negative)
    std_3 = std_negative / 2

    a2 = poc_3 + std_3
    b2 = poc_3 - std_3

    # Construct levels_data dictionary
    prices = sorted(tpo_counts.keys())
    counts = [tpo_counts[price] for price in prices]

    levels_data = {
        'prices': prices,
        'counts': counts,
        'positive_levels': {
            'Strong SELL (New High)': new_high,
            'SELL (a1)': a1,
            'POC (poc_2)': poc_2,
            'b1': b1,
            'Low (Median)': median
        },
        'negative_levels': {
            'High (Median)': median,
            'a2': a2,
            'POC (poc_3)': poc_3,
            'BUY (b2)': b2,
            'Strong BUY (New Low)': new_low
        },
        'VAH_VAL': {
            'VAH': vah,
            'VAL': val
        }
    }

    return levels_data


@csrf_exempt
def forecast(request):
    if request.method == 'POST':
        request_data = json.loads(request.body)
        stock_name = request_data.get('stockSymbol', '')  # Get the stock name from the request
        model_type = request_data['modelType']  # Get the model type from the request
        symbol2 = symbol_mapping.get(stock_name, '')  # Get the corresponding symbol
        
        if not symbol2:
            return JsonResponse({'error': 'Invalid stock symbol'})

        data = fetch_data(symbol2)
        # print(data)
        
        candlestick_data = [
                    {"x": int(date.timestamp() * 1000), "y": [open_, high, low, close]}
                    for date, open_, high, low, close in zip(
                        data.index, data['open'], data['high'], data['low'], data['close']
                    )
                ]

        if model_type == 'linear':
            forecast_data = preprocess_and_predict(data)
        elif model_type == 'random_forest':
            forecast_data = preprocess_and_predict_random_forest(data)
        elif model_type == 'ensemble':
            forecast_data = preprocess_and_predict_ensemble(data)
        else:
            return JsonResponse({'error': 'Invalid model type'})


        # Prepare candlestick data
        #candlestick_data = data[['datetime', 'open', 'high', 'low', 'close']].to_dict(orient='records')

        return JsonResponse({'forecastData': forecast_data, 'candleData': candlestick_data})
    else:
        return JsonResponse({'error': 'Invalid request method'})
    

@csrf_exempt
def result(request):
    if request.method == 'POST':
        request_data = json.loads(request.body)
        stock_name = request_data.get('stockSymbol', '')  # Get the stock name from the request
        symbol2 = symbol_mapping.get(stock_name, '')  # Get the corresponding symbol
        
        if not symbol2:
            return JsonResponse({'error': 'Invalid stock symbol'})

        data = fetch_data(symbol2)
        # print(data)
        
        candlestick_data = [
                    {"x": int(date.timestamp() * 1000), "y": [open_, high, low, close]}
                    for date, open_, high, low, close in zip(
                        data.index, data['open'], data['high'], data['low'], data['close']
                    )
                ]

        result_data = preprocess_and_predict_markovian(data)


        return JsonResponse({'resultData': result_data, 'candleData': candlestick_data})
    else:
        return JsonResponse({'error': 'Invalid request method'})
    
@csrf_exempt
def levels(request):
    if request.method == 'POST':
        request_data = json.loads(request.body)
        stock_name = request_data.get('stockSymbol', '')  # Get the stock name from the request
        symbol3 = symbol_mapping.get(stock_name, '')  # Get the corresponding symbol
        
        if not symbol3:
            return JsonResponse({'error': 'Invalid stock symbol'})

        data = fetch_data2(symbol3)
        # print(data)
        
        candlestick_data = [
                    {"x": int(date.timestamp() * 1000), "y": [open_, high, low, close]}
                    for date, open_, high, low, close in zip(
                        data.index, data['open'], data['high'], data['low'], data['close']
                    )
                ]

        levels_data = preprocess_and_calculate_levels(data)


        return JsonResponse({'levelData': levels_data, 'candleData': candlestick_data})
    else:
        return JsonResponse({'error': 'Invalid request method'})


def calculate_signals(data, fast_window=100, slow_window=400):
    data['Fast_MA'] = ta.sma(data.close, length=fast_window)
    data['Slow_MA'] = ta.sma(data.close, length=slow_window)
    data['Signal'] = np.where(data['Fast_MA'] > data['Slow_MA'], 1, -1)
    data['Position'] = data['Signal'].diff().fillna(0)
    return data

def calculate_vwap_signals(data, atr_window=14, rsi_window=14, atr_multiplier=2, profit_target_multiplier=2):
    data['VWAP'] = ta.vwap(data.high, data.low, data.close, data.volume)
    data['ATR'] = ta.atr(data.high, data.low, data.close, length=atr_window)
    data['RSI'] = ta.rsi(data.close, length=rsi_window)
    
    data['Market_Sentiment'] = np.where(data['close'] > data['VWAP'], 'Bullish', 'Bearish')
    
    # Entry Signals
    data['Entry_Point'] = np.where((data['close'].shift(1) < data['VWAP']) & (data['close'] > data['VWAP']), 'Long', 
                             np.where((data['close'].shift(1) > data['VWAP']) & (data['close'] < data['VWAP']), 'Short', 'None'))
    
    # Exit Points: Profit Target and Stop Loss
    data['Stop_Loss'] = np.where(data['Entry_Point'] == 'Long', (data['VWAP'] - (atr_multiplier * data['ATR'])), 
                            np.where(data['Entry_Point'] == 'Short', (data['VWAP'] + (atr_multiplier * data['ATR'])), None))

    data['Profit_Target'] = np.where(data['Entry_Point'] == 'Long', (data['VWAP'] + (profit_target_multiplier * data['ATR'])), 
                            np.where(data['Entry_Point'] == 'Short', (data['VWAP'] - (profit_target_multiplier * data['ATR'])), None))
    
    # Entry Point Levels
    data['Entry_Level'] = np.where(data['Entry_Point'] == 'Long', data['VWAP'], 
                            np.where(data['Entry_Point'] == 'Short', data['VWAP'], None))
    
    return data

def map_signals_ma(signal):
    if signal == 1:
        return "Buy"
    elif signal == -1:
        return "Sell"
    else:
        return "Hold"

def map_signals_vwap(entry, entry_level, stop_loss, profit_target):
    """
    Maps entry signals to trade language and displays entry level, stop loss, and profit target.
    """
    if entry == 'Long':
        return "Buy", entry_level, stop_loss, profit_target
    elif entry == 'Short':
        return "Sell", entry_level, stop_loss, profit_target
    else:
        return "None", "None", "None", "None"

# def map_trend(fast_ma, slow_ma):
#     if fast_ma > slow_ma:
#         return "Up"
#     elif fast_ma < slow_ma:
#         return "Down"
#     else:
#         return "Flat"

def map_trend(fast_ma, slow_ma):
    if fast_ma > slow_ma:
        return "Uptrend"
    elif fast_ma < slow_ma:
        return "Downtrend"
    else:
        return "Sideways"

@csrf_exempt
def strategy_result(request):
    if request.method == 'POST':
        request_data = json.loads(request.body)
        stock_name = request_data.get('stockSymbol', '')  # Get the stock name from the request
        symbol4 = symbol_mapping.get(stock_name, '')  # Get the corresponding symbol
        
        if not symbol4:
            return JsonResponse({'error': 'Invalid stock symbol'})

        data = fetch_data3(symbol4)
        
        ma_signals = calculate_signals(data.copy())
        vwap_signals = calculate_vwap_signals(data.copy())

        ma_table_data = []
        for i in range(-1, -376, -1):
            row_data = [
                str(data.index[i].date()),
                str(data.index[i].time()),
                "{:.2f}".format(data['close'].values[i]),
                "{:.2f}".format(ma_signals['Fast_MA'].values[i]),
                "{:.2f}".format(ma_signals['Slow_MA'].values[i]),
                map_signals_ma(ma_signals['Signal'].values[i]),
                map_signals_ma(ma_signals['Position'].values[i]),
                map_trend(ma_signals['Fast_MA'].values[i], ma_signals['Slow_MA'].values[i])
            ]
            ma_table_data.append(row_data)

        vwap_table_data = []
        for i in range(-1, -376, -1):
            entry_signal, entry_level, stop_loss, profit_target = map_signals_vwap(vwap_signals['Entry_Point'].values[i], vwap_signals['Entry_Level'].values[i], vwap_signals['Stop_Loss'].values[i], vwap_signals['Profit_Target'].values[i])
            entry_level = float(entry_level) if entry_level != "None" else None
            stop_loss = float(stop_loss) if stop_loss != "None" else None
            profit_target = float(profit_target) if profit_target != "None" else None
            row_data = [
                str(data.index[i].date()),
                str(data.index[i].time()),
                "{:.2f}".format(data['close'].values[i]),
                "{:.2f}".format(vwap_signals['VWAP'].values[i]),
                "{:.2f}".format(vwap_signals['ATR'].values[i]),
                "{:.2f}".format(vwap_signals['RSI'].values[i]),
                vwap_signals['Market_Sentiment'].values[i],
                entry_signal,
                "{:.2f}".format(entry_level) if entry_level is not None else "None",
                "{:.2f}".format(stop_loss) if stop_loss is not None else "None",
                "{:.2f}".format(profit_target) if profit_target is not None else "None",
                symbol4  # Include the stock symbol in each row
            ]
            vwap_table_data.append(row_data)

        return JsonResponse({
            'ma_signals': ma_table_data,
            'vwap_signals': vwap_table_data
        })
    else:
        return JsonResponse({'error': 'Invalid request method'})