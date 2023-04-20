import pandas as pd
import numpy as np
from datetime import datetime

import time
import requests

import os
from zipfile import BadZipfile



def download_url(args):
    t0 = time.time()
    url, fn = args[0], args[1]
    try:
        r = requests.get(url)
        with open(fn, 'wb') as f: 
            f.write(r.content)
        return(url, time.time() - t0)
    except Exception as e:
        print('Exception in download_url():', e)


# Upload most recent full csv

def get_num_days_since_update(init=False):
    from datetime import datetime

    if init:
        last_entry_dt = "2021-01-12 8:00:00"
    else:
        df = pd.read_csv('./../../Database/Futures_um/klines/Full_Data_klines.csv')
        df.sort_values('open_time',inplace=True)
        df.drop_duplicates(inplace=True)
        df.reset_index(drop=True,inplace=True)
        last_entry_dt = (df['open_time'].iloc[-1])

    last_entry_datetime = datetime.strptime(last_entry_dt, '%Y-%m-%d %H:%M:%S')

    import datetime
    base = datetime.datetime.today()
    num_days_since_last_entry = str(base -  last_entry_datetime).split(' ')[0]
    return num_days_since_last_entry

def get_date_list(last_entry_dt,all_dates = False):
    if all_dates:
        last_entry_dt = "2021-01-12 8:00:00"

    import datetime
    base = datetime.datetime.today()

    from datetime import datetime
    last_entry_datetime = datetime.strptime(last_entry_dt, '%Y-%m-%d %H:%M:%S')

    num_days_since_last_entry = str(base -  last_entry_datetime).split(' ')[0]

    import datetime
    date_list = [base - datetime.timedelta(days=x) for x in range(int(num_days_since_last_entry)+1)]
    return date_list

# Create functons to create the database structure

def create_Fodler_Structure():
    if len(os.listdir("./../../Database")) < 2 :
        for k in ['Futures']:    
            for i in ['um']:
                for j in ['klines']:
                    os.makedirs(f"./../../Database/{k}_{i}/{j}")
    else:
        print(f'Folder Already exists')

# Create funciton to gather urls from page and locations, and put them in a list

# Create funciton to gather urls from page and locations, and put them in a list

def get_urls_and_locations(init=False,parent_dir=['um'] , typeOfData=['klines']):
    import datetime
    if init:
        num_days_since_last_entry = get_num_days_since_update(init=True) # Seems correct!

    else:
        num_days_since_last_entry = get_num_days_since_update(init=False) # Seems correct!

    if len(num_days_since_last_entry) < 15 :
        
        date_list = [datetime.datetime.today() - datetime.timedelta(days=x) for x in range(int(num_days_since_last_entry)+1)]
        
        urls_dict={'klines':[],'aggTrades':[],'trades':[],'indexKLines':[],'MarkPriceKLines':[],'premiumIndexKLines':[],'trendingMetrics':[]}
        fns_dict = {'klines':[],'aggTerades':[],'trades':[],'indexKLines':[],'MarkPriceKLines':[],'premiumIndexKLines':[],'trendingMetrics':[]}

        parent_dir = parent_dir
        
        for i in range(len(date_list)):
            try:
                for j in parent_dir:
                    for k in typeOfData:
                        date = str(date_list[i])[:10]
                        urlk = f"https://data.binance.vision/data/futures/{j}/daily/klines/BTCBUSD/1m/BTCBUSD-1m-"+date+".zip"


                        lock = f"./../../Database/Futures_{j}/{k}/BTCBUSD-1m-"+date+".zip"
                        

                        urls_dict[k].append(urlk)
                    

                        fns_dict[k].append(lock)
            except Exception as e:
                print(e,date_list[i])
                break
        return date_list , urls_dict , fns_dict
    else:
        print("No new data to be added, try going to Yahoo Finance to updated data form today!")
        
        


# Downlaod all the data from the links
# Data form BTC daily, for klines , delays about ~10 minutes form Jan 2021 to 2023

def download_data_binance(urls_dict,fns_dict, typeOfData=['klines']):
    t0 = time.time()

    for j in typeOfData:  #['klines','aggTrades','trades','indexKLines','MarkPriceKLines','premiumIndexKLines','trendingMetrics']:
        try: 
            inputs = zip(urls_dict[j], fns_dict[j])
            for i in inputs:
                result = download_url(i)
                print('url:', result[0], 'time:', result[1])
            print('Total time:', time.time() - t0)
        except TypeError:
            continue

# MERGE

# Check if the first row is as column header and if so, fix it and add headers

def perform_sanity_checks(typeOfData=['klines']):
    for directory in typeOfData:
        li = list(os.listdir(f'./../../Database/Futures_um/klines'))
        li.sort()      
        if '.DS_Store' in li:
            li.remove('.DS_Store')
        if 'Full_Data_klines.csv' in li:
            li.remove('Full_Data_klines.csv')

        for elem in li:
            try:
                # momprint(directory , elem)
                df = pd.read_csv(f'./../../Database/Futures_um/{directory}/{elem}')
                fr = pd.DataFrame(df.columns.to_list()).T
                if fr.columns.tolist() == ['open_time'	, 'open'	, 'high', 	'low'	, 'close',	'volume'	,'close_time'	,'quote_volume'	,'count',	'taker_buy_volume'	,'taker_buy_quote_volume'	,'ignore']:
                    print(f"{elem} doesnt change")
                else:
                    fr.columns = ['open_time'	, 'open'	, 'high', 	'low'	, 'close',	'volume'	,'close_time'	,'quote_volume'	,'count',	'taker_buy_volume'	,'taker_buy_quote_volume'	,'ignore']
                    df.columns = ['open_time'	, 'open'	, 'high', 	'low'	, 'close',	'volume'	,'close_time'	,'quote_volume'	,'count',	'taker_buy_volume'	,'taker_buy_quote_volume'	,'ignore']
                    df_conc = pd.concat([fr,df])
                    df_conc = df_conc.loc[(df_conc['open_time'] != 'open_time')]
                    df_conc.to_csv(f'./../../Database/Futures_um/{directory}/{elem}',index=False,compression='zip')
            except BadZipfile:
                print(f"{directory , elem} doesnt exist yet ")
                continue
            except Exception as e:
                print(e , elem)
                continue

# Create quickly num Sanity Check

def floats_sanity_check(df):
    for col in ['high', 'low', 'open', 'close', 'taker_buy_volume', 'taker_buy_quote_volume']:
        for row in range(df.shape[0]):
            try:
                if len(df[col].loc[row].split('.')) > 2:
                    df[col].loc[row] = '.'.join(df[col].loc[row].split('.')[:2])
            except Exception as e:
                continue
    return df

def merge_into_single_csv(date_list,init=False,typeOfData=['klines']):    
    # do a mega csv of each of te folder components
    for directory in typeOfData:
        try:
            if init:
                base_df = pd.DataFrame(columns=['open_time'	, 'open'	, 'high', 	'low'	, 'close',	'volume'	,'close_time'	,'quote_volume'	,'count',	'taker_buy_volume'	,'taker_buy_quote_volume'	,'ignore'])
            else:
                base_df = pd.read_csv('./../../Database/Futures_um/klines/Full_Data_klines.csv')

            for i in range(len(date_list)):
                #print(range(len(date_list)))
                dat = str(date_list[i])[:10]
                link = f'./../../Database/Futures_um/{directory}/BTCBUSD-1m-{dat}.zip'
                #print(dat)
                try :
                    df = pd.read_csv(link)
                    base_df = pd.concat([base_df , df])
                except BadZipfile:
                    continue
                except Exception as e:
                    print(e , date_list[i])
                    continue
                
            base_df.reset_index(drop=True,inplace=True)
            base_df['open_time'] = base_df['open_time'].apply(lambda x: x/1)

            #base_df['open_time']=pd.to_datetime(base_df['open_time'],unit='ns')
            base_df['open_time']=pd.to_datetime(base_df['open_time'],unit='ms')

            base_df.sort_values('open_time',inplace=True)
            base_df.reset_index(drop=True,inplace=True)

            merged_df = floats_sanity_check(base_df)

            merged_df.drop_duplicates(inplace=True)
            
            merged_df.to_csv(f'./../../Database/Futures_um/{directory}/Full_Data_klines.csv',index=False)
            print(f'completed klines')
        except Exception as e:
            print(e,dat)
            continue
    
        #return merged_df

def create_Stream(init=False,redownload=False):
    if init==True:
        
        date_list , urls_dict , fns_dict = get_urls_and_locations(init=init)
        download_data_binance(urls_dict , fns_dict)
        perform_sanity_checks()
        merge_into_single_csv(date_list)

    else:
        #create_Fodler_Structure()
        
        try:
            if redownload:
                date_list , urls_dict , fns_dict = get_urls_and_locations(init=init)
                download_data_binance(urls_dict , fns_dict)
                perform_sanity_checks()
                merge_into_single_csv(date_list)
            else:
                date_list = get_date_list("2021-01-12" , all_dates=True)
                perform_sanity_checks()
                merge_into_single_csv(date_list)
        except Exception as e:
            print(e)


create_Stream(init=False, redownload=False)




