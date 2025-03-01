import pandas as pd

def clean_data_runs(df):
    df.drop(columns=['finish_time', 'horse_no',
                     'position_sec1', 'position_sec2', 'position_sec3', 'position_sec4', 'position_sec5',
                     'position_sec6',
                     'behind_sec1', 'behind_sec2', 'behind_sec3', 'behind_sec4', 'behind_sec5', 'behind_sec6',
                     'time1', 'time2', 'time3', 'time4', 'time5', 'time6'], inplace=True)

def clean_data_races(df):
    df.drop(columns=['sec_time1', 'sec_time2', 'sec_time3', 'sec_time4', 'sec_time5', 'sec_time6', 'sec_time7',
                     'time1', 'time2', 'time3', 'time4', 'time5', 'time6', 'time7'
                     ], inplace=True)