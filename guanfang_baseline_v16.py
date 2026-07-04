#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AFAC2026 赛题一 完整流水线 v14 — 天池标准格式版
=============================================
基于 guanfang_baseline_v14.py，仅修改输出格式以匹配天池要求：
 - pattern_reco.csv: stock_code, transaction_date, pattern_id (1-8整数)
 - predict_result.csv: stock_code, transaction_date, intent_id (1-3整数)
核心逻辑（特征提取、聚类、XGBoost分类）完全保留v14。
"""

import glob
import pandas as pd
import numpy as np
import json
import argparse
import os
import sys
import warnings
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.optimize import linear_sum_assignment
from collections import Counter

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

warnings.filterwarnings('ignore')

# ============================================================
# PART 1: 数据生成（来自 generate_afac_train_v3.py）
# ============================================================

TARGET_COLUMNS = [
    'channelexchange', 'channellevel', 'seqnum', 'tradechannel', 'symbol',
    'date', 'tradedate', 'name', 'price', 'openprice', 'highprice', 'lowprice',
    'closeprice', 'previousclose', 'averageprice', 'changeprice', 'changepercent',
    'rangepercent', 'volume', 'currentvolume', 'amount', 'transactions',
    'volumerate', 'upprice', 'downprice', 'bidaskrate', 'bidaskdifference',
    'weightedbidprice', 'weightedaskprice', 'totalbidvolume', 'totalaskvolume',
    'iopv', 'iopvtm1', 'tradestate', 'suspensionstate', 'snapshotdate',
    'tradetimeoverrideprotocol', 'pricepublishtype', 'aftermarketprice',
    'aftermarketbuyvolume', 'aftermarketsellvolume', 'aftermarketvolume',
    'aftermarketamount', 'aftermarkettradingphasecode', 'aftermarkettradetime',
    'bids', 'asks', 'bigordervolume', 'position', 'limitstate',
    'aftermarkettransactions', 'aftermarketbuyorders', 'aftermarketsellorders',
    'premiumrate', 'subtradingphases', 'auctionvolume', 'auctionamount',
    'auctionprice', 'ticksize', 'minbuylimitprice', 'maxbuylimitprice',
    'minselllimitprice', 'maxselllimitprice', 'dt', 'hh'
]

PRICE_FIELDS = ['price', 'openprice', 'highprice', 'lowprice', 'closeprice',
    'previousclose', 'averageprice', 'changeprice',
    'weightedbidprice', 'weightedaskprice',
    'upprice', 'downprice',
    'minbuylimitprice', 'maxbuylimitprice',
    'minselllimitprice', 'maxselllimitprice',
    'aftermarketprice', 'auctionprice', 'ticksize']


def read_csv_robust(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")
    for enc in ['utf-8', 'gbk', 'gb2312', 'gb18030', 'cp936']:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path, encoding='utf-8', errors='ignore')


def parse_time_to_ms_timestamp(date_int, time_int):
    try:
        d = str(int(date_int))
        year, month, day = int(d[:4]), int(d[4:6]), int(d[6:8])
        t = str(int(time_int)).zfill(9)
        hh, mm, ss = int(t[:2]), int(t[2:4]), int(t[4:6])
        mmm = int(t[6:9]) if len(t) > 6 else 0
        dt = datetime(year, month, day, hh, mm, ss, mmm * 1000)
        dt_utc = dt - timedelta(hours=8)
        return int(dt_utc.timestamp() * 1000)
    except Exception:
        return 0


def merge_bid_ask_to_json(df):
    bid_price_cols, bid_vol_cols, ask_price_cols, ask_vol_cols = [], [], [], []
    for col in df.columns:
        c = str(col).strip()
        if c.startswith('申买价') and c[3:].isdigit():
            bid_price_cols.append((int(c[3:]), col))
        elif c.startswith('申买量') and c[3:].isdigit():
            bid_vol_cols.append((int(c[3:]), col))
        elif c.startswith('申卖价') and c[3:].isdigit():
            ask_price_cols.append((int(c[3:]), col))
        elif c.startswith('申卖量') and c[3:].isdigit():
            ask_vol_cols.append((int(c[3:]), col))

    for lst in [bid_price_cols, bid_vol_cols, ask_price_cols, ask_vol_cols]:
        lst.sort(key=lambda x: x[0])

    def build_bids(row):
        bids = []
        for idx, (lvl, pcol) in enumerate(bid_price_cols):
            vcol = bid_vol_cols[idx][1] if idx < len(bid_vol_cols) else None
            p, v = row.get(pcol, 0), row.get(vcol, 0) if vcol else 0
            if pd.notna(p) and pd.notna(v) and int(v) > 0 and int(p) > 0:
                bids.append({'price': int(p) / 10000.0, 'volume': int(v),
                    'order': [{'volume': int(v)}], 'bigOrderPercent': 0.0})
        return json.dumps(bids, ensure_ascii=False, separators=(',', ':'))

    def build_asks(row):
        asks = []
        for idx, (lvl, pcol) in enumerate(ask_price_cols):
            vcol = ask_vol_cols[idx][1] if idx < len(ask_vol_cols) else None
            p, v = row.get(pcol, 0), row.get(vcol, 0) if vcol else 0
            if pd.notna(p) and pd.notna(v) and int(v) > 0 and int(p) > 0:
                asks.append({'price': int(p) / 10000.0, 'volume': int(v),
                    'order': [{'volume': int(v)}], 'bigOrderPercent': 0.0})
        return json.dumps(asks, ensure_ascii=False, separators=(',', ':'))

    if bid_price_cols and bid_vol_cols:
        df['bids'] = df.apply(build_bids, axis=1)
    if ask_price_cols and ask_vol_cols:
        df['asks'] = df.apply(build_asks, axis=1)

    cols_to_drop = [c[1] for c in bid_price_cols + bid_vol_cols + ask_price_cols + ask_vol_cols]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
    return df


def map_wind_snapshot(df):
    wind_map = {
        '万得代码': 'symbol', '交易所代码': 'exchange_code', '自然日': 'date', '时间': 'time',
        '成交价': 'price', '成交量': 'currentvolume', '成交额': 'tick_amount',
        '成交笔数': 'transactions', 'IOPV': 'iopv', '成交标志': 'trade_flag',
        'BS标志': 'bs_flag', '当日累计成交量': 'volume', '当日成交额': 'amount',
        '最高价': 'highprice', '最低价': 'lowprice', '开盘价': 'openprice',
        '前收盘': 'previousclose', '加权平均叫买价': 'weightedbidprice',
        '加权平均叫卖价': 'weightedaskprice', '叫买总量': 'totalbidvolume',
        '叫卖总量': 'totalaskvolume', '不加权指数': 'unweighted_index',
        '品种总数': 'variety_count', '上涨品种数': 'up_count',
        '下跌品种数': 'down_count', '持平品种数': 'flat_count',
    }
    rename_map = {k: v for k, v in wind_map.items() if k in df.columns}
    df = df.rename(columns=rename_map)
    unnamed_cols = [c for c in df.columns if 'Unnamed' in str(c)]
    df = df.drop(columns=unnamed_cols, errors='ignore')
    return df


def map_wind_trade(df):
    wind_map = {
        '万得代码': 'symbol', '交易所代码': 'name', '自然日': 'date', '时间': 'time',
        '成交编号': 'trade_no', '成交代码': 'trade_code', '委托代码': 'order_code',
        'BS标志': 'bs', '成交价格': 'price', '成交数量': 'volume',
        '叫卖序号': 'ask_order_no', '叫买序号': 'bid_order_no',
    }
    rename_map = {k: v for k, v in wind_map.items() if k in df.columns}
    return df.rename(columns=rename_map)


def map_wind_order(df):
    wind_map = {
        '万得代码': 'symbol', '交易所代码': 'name', '自然日': 'date', '时间': 'time',
        '委托编号': 'order_no', '交易所委托号': 'ex_order_no', '委托类型': 'order_type',
        '委托代码': 'order_code', '委托价格': 'price', '委托数量': 'volume',
    }
    rename_map = {k: v for k, v in wind_map.items() if k in df.columns}
    return df.rename(columns=rename_map)


def convert_units(df):
    for col in PRICE_FIELDS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce') / 10000.0
    return df


def compute_derived(df, stock_name):
    """计算派生字段，修复 dt 日期格式（Bug 4）"""
    orig_date = int(df['date'].iloc[0]) if 'date' in df.columns and len(df) > 0 else 0

    if stock_name:
        df['name'] = stock_name
    if 'closeprice' not in df.columns or df['closeprice'].isna().all():
        df['closeprice'] = df['price']
    if 'changeprice' not in df.columns or df['changeprice'].isna().all():
        df['changeprice'] = df['price'] - df['previousclose']
    if 'changepercent' not in df.columns or df['changepercent'].isna().all():
        mask = df['previousclose'] != 0
        df.loc[mask, 'changepercent'] = (df.loc[mask, 'changeprice'] / df.loc[mask, 'previousclose'] * 100).round(2)
        df.loc[~mask, 'changepercent'] = 0.0
    if 'rangepercent' not in df.columns or df['rangepercent'].isna().all():
        range_val = df['highprice'] - df['lowprice']
        mask = df['previousclose'] != 0
        df.loc[mask, 'rangepercent'] = (range_val[mask] / df.loc[mask, 'previousclose'] * 100).round(2)
        df.loc[~mask, 'rangepercent'] = 0.0
    if 'averageprice' not in df.columns or df['averageprice'].isna().all():
        mask = df['volume'] != 0
        df.loc[mask, 'averageprice'] = (df.loc[mask, 'amount'] / df.loc[mask, 'volume']).round(2)
        df.loc[~mask, 'averageprice'] = df['price']
    if 'upprice' not in df.columns or df['upprice'].isna().all():
        df['upprice'] = (df['previousclose'] * 1.1).round(2)
    if 'downprice' not in df.columns or df['downprice'].isna().all():
        df['downprice'] = (df['previousclose'] * 0.9).round(2)
    for col, val in [('minbuylimitprice', 'downprice'), ('maxbuylimitprice', 'upprice'),
        ('minselllimitprice', 'downprice'), ('maxselllimitprice', 'upprice')]:
        if col not in df.columns or df[col].isna().all():
            df[col] = df[val]
    if 'dt' not in df.columns or df['dt'].isna().all():
        df['dt'] = orig_date
    if 'hh' not in df.columns or df['hh'].isna().all():
        df['hh'] = df['time'].apply(lambda x: int(str(int(x)).zfill(9)[:2]) if pd.notna(x) else 0)
    if 'date' in df.columns and 'time' in df.columns:
        df['date'] = df.apply(lambda row: parse_time_to_ms_timestamp(row['date'], row['time']), axis=1)
    if 'seqnum' not in df.columns or df['seqnum'].isna().all():
        df['seqnum'] = range(len(df))
    if 'channelexchange' not in df.columns or df['channelexchange'].isna().all():
        df['channelexchange'] = df['symbol'].apply(lambda x: 'SH' if '.SH' in str(x) else ('SZ' if '.SZ' in str(x) else 'SH'))
    if 'channellevel' not in df.columns or df['channellevel'].isna().all():
        df['channellevel'] = 1
    if 'tradechannel' not in df.columns or df['tradechannel'].isna().all():
        df['tradechannel'] = 1
    if 'tradedate' not in df.columns or df['tradedate'].isna().all():
        df['tradedate'] = int(df['dt'].iloc[0]) if len(df) > 0 else 0
    if 'ticksize' not in df.columns or df['ticksize'].isna().all():
        df['ticksize'] = 0.01
    for col in ['tradestate', 'suspensionstate', 'limitstate', 'position', 'iopvtm1',
        'volumerate', 'snapshotdate', 'tradetimeoverrideprotocol', 'pricepublishtype']:
        if col not in df.columns or df[col].isna().all():
            df[col] = 0
    for col in ['aftermarketprice', 'aftermarketbuyvolume', 'aftermarketsellvolume',
        'aftermarketvolume', 'aftermarketamount', 'aftermarkettradingphasecode',
        'aftermarkettradetime', 'aftermarkettransactions', 'aftermarketbuyorders',
        'aftermarketsellorders', 'premiumrate', 'subtradingphases',
        'auctionvolume', 'auctionamount', 'auctionprice']:
        if col not in df.columns or df[col].isna().all():
            df[col] = 0
    return df


def supplement_from_trade(df, df_trade, big_threshold=10000):
    """从逐笔成交补充字段，修复 bigordervolume 按时间窗口聚合（Bug 5）"""
    if df_trade is None or df_trade.empty:
        return df
    df_trade = map_wind_trade(df_trade)
    df['symbol'] = df['symbol'].astype(str)
    df_trade['symbol'] = df_trade['symbol'].astype(str)

    if 'time' in df_trade.columns:
        df_trade['time_norm'] = df_trade['time'].apply(lambda x: int(str(int(x)).zfill(9)[:6]) if pd.notna(x) else 0)
    if 'time' in df.columns:
        df['time_norm'] = df['time'].apply(lambda x: int(str(int(x)).zfill(9)[:6]) if pd.notna(x) else 0)

    if 'bigordervolume' not in df.columns or df['bigordervolume'].isna().all():
        big_orders = df_trade[df_trade['volume'] >= big_threshold].groupby(['symbol', 'time_norm'])['volume'].sum().reset_index(name='bigvol')
        df = df.merge(big_orders, left_on=['symbol', 'time_norm'], right_on=['symbol', 'time_norm'], how='left')
        df['bigordervolume'] = df['bigvol'].fillna(0).astype(int)
        df.drop(columns=['bigvol'], inplace=True, errors='ignore')

    if 'bs' in df_trade.columns and 'time' in df_trade.columns:
        buy_vol = df_trade[df_trade['bs'].astype(str).str.upper().isin(['B', 'BUY', '买'])]
        sell_vol = df_trade[df_trade['bs'].astype(str).str.upper().isin(['S', 'SELL', '卖'])]
        buy_agg = buy_vol.groupby(['symbol', 'time_norm'])['volume'].sum().reset_index(name='buy_vol')
        sell_agg = sell_vol.groupby(['symbol', 'time_norm'])['volume'].sum().reset_index(name='sell_vol')
        df = df.merge(buy_agg, left_on=['symbol', 'time_norm'], right_on=['symbol', 'time_norm'], how='left')
        df = df.merge(sell_agg, left_on=['symbol', 'time_norm'], right_on=['symbol', 'time_norm'], how='left')
        df['buy_vol'] = df['buy_vol'].fillna(0)
        df['sell_vol'] = df['sell_vol'].fillna(0)
        if 'bidaskdifference' not in df.columns or df['bidaskdifference'].isna().all():
            df['bidaskdifference'] = (df['buy_vol'] - df['sell_vol']).astype(int)
        if 'bidaskrate' not in df.columns or df['bidaskrate'].isna().all():
            total = df['buy_vol'] + df['sell_vol']
            mask = total > 0
            df.loc[mask, 'bidaskrate'] = (df.loc[mask, 'buy_vol'] / total[mask]).round(6)
            df.loc[~mask, 'bidaskrate'] = 0.5
        df.drop(columns=['buy_vol', 'sell_vol', 'time_norm'], inplace=True, errors='ignore')
    return df


def supplement_from_order(df, df_order):
    """从逐笔委托补充字段，修复撤单统计按 symbol 聚合导致的常量问题（Bug 2/3）"""
    if df_order is None or df_order.empty:
        return df
    df_order = map_wind_order(df_order)
    df['symbol'] = df['symbol'].astype(str)
    df_order['symbol'] = df_order['symbol'].astype(str)

    if 'time' in df_order.columns:
        df_order['time_norm'] = df_order['time'].apply(lambda x: int(str(int(x)).zfill(9)[:6]) if pd.notna(x) else 0)
    if 'time' in df.columns:
        df['time_norm'] = df['time'].apply(lambda x: int(str(int(x)).zfill(9)[:6]) if pd.notna(x) else 0)

    if 'order_code' in df_order.columns:
        buy_orders = df_order[df_order['order_code'].astype(str).str.upper().isin(['B', 'BUY', '买'])]
        sell_orders = df_order[df_order['order_code'].astype(str).str.upper().isin(['S', 'SELL', '卖'])]
    elif 'order_type' in df_order.columns:
        buy_orders = df_order[df_order['order_type'].astype(str).str.upper().isin(['B', 'BUY', '买'])]
        sell_orders = df_order[df_order['order_type'].astype(str).str.upper().isin(['S', 'SELL', '卖'])]
    else:
        return df

    cancel_mask = None
    detected_method = None

    if 'order_type' in df_order.columns:
        ot = df_order['order_type'].astype(str).str.upper()
        m1 = ot == 'D'
        if m1.any():
            cancel_mask = m1
            detected_method = 'order_type=D'

    if cancel_mask is None and 'order_type' in df_order.columns:
        ot = df_order['order_type'].astype(str).str.upper()
        m1b = ot.isin(['U', 'CANCEL', '撤', 'DELETE', '3', '4'])
        if m1b.any():
            cancel_mask = m1b
            detected_method = 'order_type_fallback'

    if cancel_mask is None and 'trade_code' in df_order.columns:
        tc = df_order['trade_code'].astype(str).str.upper()
        m3 = tc == 'C'
        if m3.any():
            cancel_mask = m3
            detected_method = 'trade_code=C'

    if cancel_mask is None and 'order_code' in df_order.columns:
        oc = df_order['order_code'].astype(str).str.upper()
        m2 = oc.isin(['C', 'CANCEL', '撤', 'DELETE'])
        if m2.any():
            cancel_mask = m2
            detected_method = 'order_code'

    cancel_orders = df_order[cancel_mask] if cancel_mask is not None else pd.DataFrame()

    if not cancel_orders.empty:
        print(f"  检测到撤单记录: {len(cancel_orders)}条 (method={detected_method})")

    cancel_buy = cancel_orders.copy()
    cancel_sell = cancel_orders.copy()

    if 'order_code' in cancel_orders.columns:
        coc = cancel_orders['order_code'].astype(str).str.upper()
        cancel_buy = cancel_orders[coc == 'B']
        cancel_sell = cancel_orders[coc == 'S']
    elif 'order_type' in cancel_orders.columns:
        cot = cancel_orders['order_type'].astype(str).str.upper()
        cancel_buy = cancel_orders[cot == 'B']
        cancel_sell = cancel_orders[cot == 'S']

    cancel_stats = cancel_orders.groupby('symbol').agg({'volume': ['sum', 'count']}).reset_index()
    cancel_stats.columns = ['symbol', 'cancel_volume', 'cancel_count']

    cb_buy_stats = cancel_buy.groupby('symbol')['volume'].sum().reset_index(name='cancel_buy_volume')
    cb_sell_stats = cancel_sell.groupby('symbol')['volume'].sum().reset_index(name='cancel_sell_volume')
    cb_buy_count_stats = cancel_buy.groupby('symbol').size().reset_index(name='cancel_buy_count')
    cb_sell_count_stats = cancel_sell.groupby('symbol').size().reset_index(name='cancel_sell_count')

    cancel_interval_cv = 0.0
    if 'time' in cancel_orders.columns:
        cancel_orders['time_val'] = pd.to_numeric(cancel_orders['time'], errors='coerce')
        cancel_sorted = cancel_orders.sort_values(['symbol', 'time_val'])
        cancel_sorted['cancel_interval_ms'] = cancel_sorted.groupby('symbol')['time_val'].diff().abs()
        valid_intervals = cancel_sorted['cancel_interval_ms'].dropna()
        if len(valid_intervals) > 1 and valid_intervals.mean() > 0:
            cancel_interval_cv = valid_intervals.std() / valid_intervals.mean()
        fast_cancel = cancel_sorted[cancel_sorted['cancel_interval_ms'] < 1000]
        fast_stats = fast_cancel.groupby('symbol').size().reset_index(name='fast_cancel_count')
    else:
        fast_stats = pd.DataFrame(columns=['symbol', 'fast_cancel_count'])

    df = df.merge(cancel_stats, on='symbol', how='left')
    df = df.merge(cb_buy_stats, on='symbol', how='left')
    df = df.merge(cb_sell_stats, on='symbol', how='left')
    df = df.merge(cb_buy_count_stats, on='symbol', how='left')
    df = df.merge(cb_sell_count_stats, on='symbol', how='left')
    df = df.merge(fast_stats, on='symbol', how='left')

    for col in ['cancel_volume', 'cancel_count', 'cancel_buy_volume', 'cancel_sell_volume', 'cancel_buy_count', 'cancel_sell_count', 'fast_cancel_count']:
        df[col] = df[col].fillna(0).astype(int)
    df['cancel_interval_cv'] = cancel_interval_cv

    if 'volume' in df_order.columns:
        order_max = df_order.groupby('symbol')['volume'].max().reset_index(name='order_max_volume')
        df = df.merge(order_max, on='symbol', how='left')
        df['order_max_volume'] = df['order_max_volume'].fillna(0).astype(int)
        order_mega = df_order[df_order['volume'] >= 50000].groupby('symbol').size().reset_index(name='order_mega_count')
        df = df.merge(order_mega, on='symbol', how='left')
        df['order_mega_count'] = df['order_mega_count'].fillna(0).astype(int)
        order_large = df_order[df_order['volume'] >= 10000].groupby('symbol').size().reset_index(name='order_large_count')
        df = df.merge(order_large, on='symbol', how='left')
        df['order_large_count'] = df['order_large_count'].fillna(0).astype(int)
    else:
        df['order_max_volume'] = 0
        df['order_mega_count'] = 0
        df['order_large_count'] = 0

    for col in ['cancel_volume', 'cancel_count', 'cancel_buy_volume', 'cancel_sell_volume', 'cancel_buy_count', 'cancel_sell_count', 'fast_cancel_count']:
        if col not in df.columns:
            df[col] = 0
    if 'cancel_interval_cv' not in df.columns:
        df['cancel_interval_cv'] = 0.0

    wbp = buy_orders.groupby('symbol').apply(
        lambda x: (x['price'] * x['volume']).sum() / x['volume'].sum() if x['volume'].sum() > 0 else 0
    ).reset_index(name='wbp')
    df = df.merge(wbp, on='symbol', how='left')
    if 'weightedbidprice' not in df.columns or df['weightedbidprice'].isna().all():
        df['weightedbidprice'] = df['wbp'].fillna(0).round(2)
    df.drop(columns=['wbp'], inplace=True, errors='ignore')

    if not sell_orders.empty:
        wap = sell_orders.groupby('symbol').apply(
            lambda x: (x['price'] * x['volume']).sum() / x['volume'].sum() if x['volume'].sum() > 0 else 0
        ).reset_index(name='wap')
        df = df.merge(wap, on='symbol', how='left')
        if 'weightedaskprice' not in df.columns or df['weightedaskprice'].isna().all():
            df['weightedaskprice'] = df['wap'].fillna(0).round(2)
        df.drop(columns=['wap'], inplace=True, errors='ignore')

    buy_vol = buy_orders.groupby('symbol')['volume'].sum().reset_index(name='bid_vol')
    sell_vol = sell_orders.groupby('symbol')['volume'].sum().reset_index(name='ask_vol')
    df = df.merge(buy_vol, on='symbol', how='left')
    df = df.merge(sell_vol, on='symbol', how='left')
    if 'totalbidvolume' not in df.columns or df['totalbidvolume'].isna().all():
        df['totalbidvolume'] = df['bid_vol'].fillna(0).astype(int)
    if 'totalaskvolume' not in df.columns or df['totalaskvolume'].isna().all():
        df['totalaskvolume'] = df['ask_vol'].fillna(0).astype(int)
    df.drop(columns=['bid_vol', 'ask_vol'], inplace=True, errors='ignore')

    if 'bidaskrate' not in df.columns or df['bidaskrate'].isna().all():
        total = df['totalbidvolume'] + df['totalaskvolume']
        mask = total > 0
        df.loc[mask, 'bidaskrate'] = (df.loc[mask, 'totalbidvolume'] / total[mask]).round(6)
        df.loc[~mask, 'bidaskrate'] = 0.5
    if 'bidaskdifference' not in df.columns or df['bidaskdifference'].isna().all():
        df['bidaskdifference'] = (df['totalbidvolume'] - df['totalaskvolume']).astype(int)
    return df


def fill_and_reorder(df):
    defaults = {
        'channelexchange': 'SH', 'channellevel': 'L2', 'seqnum': 0, 'tradechannel': None,
        'symbol': '', 'date': 0, 'tradedate': None, 'name': '', 'price': 0.0,
        'openprice': None, 'highprice': None, 'lowprice': None, 'closeprice': None,
        'previousclose': 0.0, 'averageprice': None, 'changeprice': 0.0, 'changepercent': 0.0,
        'rangepercent': 0.0, 'volume': 0, 'currentvolume': 0, 'amount': 0.0,
        'transactions': 0, 'volumerate': 0, 'upprice': 0.0, 'downprice': 0.0,
        'bidaskrate': 0, 'bidaskdifference': 0, 'weightedbidprice': None,
        'weightedaskprice': None, 'totalbidvolume': 0, 'totalaskvolume': 0,
        'iopv': None, 'iopvtm1': None, 'tradestate': 0, 'suspensionstate': 1,
        'snapshotdate': 0, 'tradetimeoverrideprotocol': None, 'pricepublishtype': 0,
        'aftermarketprice': None, 'aftermarketbuyvolume': 0, 'aftermarketsellvolume': 0,
        'aftermarketvolume': 0, 'aftermarketamount': 0, 'aftermarkettradingphasecode': 0,
        'aftermarkettradetime': None, 'bids': '[]', 'asks': '[]', 'bigordervolume': 0,
        'position': 0, 'limitstate': 0, 'aftermarkettransactions': 0,
        'aftermarketbuyorders': '[]', 'aftermarketsellorders': '[]', 'premiumrate': None,
        'subtradingphases': '[]', 'auctionvolume': 0, 'auctionamount': 0,
        'auctionprice': 0, 'ticksize': 1, 'minbuylimitprice': 0, 'maxbuylimitprice': 0,
        'minselllimitprice': 0, 'maxselllimitprice': 0, 'dt': 0, 'hh': 0
    }
    for col in TARGET_COLUMNS:
        if col not in df.columns:
            df[col] = defaults.get(col, 0)
        else:
            df[col] = df[col].fillna(defaults.get(col, 0))

    int_cols = ['seqnum', 'volume', 'currentvolume', 'transactions', 'totalbidvolume',
        'totalaskvolume', 'bigordervolume', 'position', 'aftermarketvolume',
        'aftermarketamount', 'aftermarkettransactions', 'aftermarketbuyvolume',
        'aftermarketsellvolume', 'auctionvolume', 'auctionamount', 'dt', 'hh',
        'bidaskdifference', 'tradestate', 'suspensionstate', 'snapshotdate',
        'aftermarkettradingphasecode', 'limitstate', 'premiumrate', 'auctionprice',
        'ticksize', 'upprice', 'downprice', 'minbuylimitprice', 'maxbuylimitprice',
        'minselllimitprice', 'maxselllimitprice', 'pricepublishtype']
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    float_cols = ['price', 'openprice', 'highprice', 'lowprice', 'closeprice', 'previousclose',
        'averageprice', 'changeprice', 'changepercent', 'rangepercent',
        'weightedbidprice', 'weightedaskprice', 'amount', 'aftermarketprice',
        'aftermarkettradetime', 'tradetimeoverrideprotocol', 'iopv', 'iopvtm1']
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    extra_cols = [c for c in df.columns if c not in TARGET_COLUMNS]
    df = df[TARGET_COLUMNS + extra_cols]
    return df


def process_single_stock(snapshot_path, trade_path, order_path, output_path,
    stock_name='', big_threshold=10000):
    df_snap = read_csv_robust(snapshot_path)
    df_trade = read_csv_robust(trade_path)
    df_order = read_csv_robust(order_path)

    df_snap = map_wind_snapshot(df_snap)
    df_snap = merge_bid_ask_to_json(df_snap)
    df_snap = convert_units(df_snap)
    df_snap = supplement_from_trade(df_snap, df_trade, big_threshold=big_threshold)
    df_snap = supplement_from_order(df_snap, df_order)
    df_snap = compute_derived(df_snap, stock_name=stock_name)
    df_result = fill_and_reorder(df_snap)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    df_result.to_excel(output_path, index=False, engine='openpyxl')
    return len(df_result)


def generate_train_data(batch_dir, output_dir, merge=False, big_threshold=10000):
    print(f"\n{'='*65}")
    print(f"[阶段1/2] 生成训练数据")
    print(f"{'='*65}")

    if not os.path.exists(batch_dir):
        raise FileNotFoundError(f"输入目录不存在: {batch_dir}")

    os.makedirs(output_dir, exist_ok=True)
    total_rows, success_count, fail_count = 0, 0, 0

    subdirs = sorted([d for d in os.listdir(batch_dir)
        if os.path.isdir(os.path.join(batch_dir, d))])

    is_double_layer = False
    if subdirs:
        first_sub = os.path.join(batch_dir, subdirs[0])
        sub_subdirs = [d for d in os.listdir(first_sub)
            if os.path.isdir(os.path.join(first_sub, d))]
        if sub_subdirs:
            test_path = os.path.join(first_sub, sub_subdirs[0], '行情.csv')
            is_double_layer = os.path.exists(test_path)

    if is_double_layer:
        print(f"检测到双层目录结构（日期/股票）: {len(subdirs)} 个交易日")
        all_stock_dirs = []
        for date_dir in subdirs:
            date_path = os.path.join(batch_dir, date_dir)
            stock_dirs = sorted([d for d in os.listdir(date_path)
                if os.path.isdir(os.path.join(date_path, d))])
            for stock in stock_dirs:
                all_stock_dirs.append((date_dir, stock, os.path.join(date_path, stock)))
        print(f" 共 {len(all_stock_dirs)} 个股票-日期组合\n")

        for i, (date_dir, code, stock_path) in enumerate(all_stock_dirs, 1):
            snapshot_path = os.path.join(stock_path, '行情.csv')
            trade_path = os.path.join(stock_path, '逐笔成交.csv')
            order_path = os.path.join(stock_path, '逐笔委托.csv')
            date_output_dir = os.path.join(output_dir, date_dir)
            os.makedirs(date_output_dir, exist_ok=True)
            output_path = os.path.join(date_output_dir, f'{code}.xlsx')

            missing = [label for label, path in [('行情', snapshot_path), ('逐笔成交', trade_path), ('逐笔委托', order_path)]
                if not os.path.exists(path)]
            if missing:
                print(f"[{i}/{len(all_stock_dirs)}] ⚠️ {date_dir}/{code} 跳过 — 缺失: {', '.join(missing)}")
                fail_count += 1
                continue

            print(f"[{i}/{len(all_stock_dirs)}] 📦 {date_dir}/{code} 处理中...", end=' ')
            try:
                rows = process_single_stock(snapshot_path, trade_path, order_path, output_path,
                    stock_name=code, big_threshold=big_threshold)
                total_rows += rows
                success_count += 1
                print(f"✅ {rows}行")
            except Exception as e:
                fail_count += 1
                print(f"❌ 失败: {e}")
    else:
        print(f"发现 {len(subdirs)} 个股票目录\n")
        for i, code in enumerate(subdirs, 1):
            stock_dir = os.path.join(batch_dir, code)
            snapshot_path = os.path.join(stock_dir, '行情.csv')
            trade_path = os.path.join(stock_dir, '逐笔成交.csv')
            order_path = os.path.join(stock_dir, '逐笔委托.csv')
            output_path = os.path.join(output_dir, f'{code}.xlsx')

            missing = [label for label, path in [('行情', snapshot_path), ('逐笔成交', trade_path), ('逐笔委托', order_path)]
                if not os.path.exists(path)]
            if missing:
                print(f"[{i}/{len(subdirs)}] ⚠️ {code} 跳过 — 缺失: {', '.join(missing)}")
                fail_count += 1
                continue

            print(f"[{i}/{len(subdirs)}] 📦 {code} 处理中...", end=' ')
            try:
                rows = process_single_stock(snapshot_path, trade_path, order_path, output_path,
                    stock_name=code, big_threshold=big_threshold)
                total_rows += rows
                success_count += 1
                print(f"✅ {rows}行")
            except Exception as e:
                fail_count += 1
                print(f"❌ 失败: {e}")

    print(f"\n{'='*65}")
    print(f"数据生成完成: 成功 {success_count} / 失败 {fail_count} / 总计 {success_count + fail_count}")
    print(f"总数据行: {total_rows}")

    if merge and success_count > 0:
        merge_path = os.path.join(output_dir, 'all_100.csv')
        xlsx_files = sorted(glob.glob(os.path.join(output_dir, '**/*.xlsx'), recursive=True))
        xlsx_files = [f for f in xlsx_files if os.path.basename(f) != 'all_100.xlsx']

        if xlsx_files:
            print(f"\n[*] 合并 {len(xlsx_files)} 个文件 → all_100.csv...")
            # 分块写入 CSV，避免 openpyxl 内存爆炸
            first = True
            for f in xlsx_files:
                try:
                    chunk = pd.read_excel(f)
                    chunk.to_csv(merge_path, index=False, mode='w' if first else 'a', header=first, encoding='utf-8-sig')
                    first = False
                except Exception as e:
                    print(f" ⚠️ 跳过 {os.path.basename(f)}: {e}")
            if not first:
                print(f"✅ 合并完成: {merge_path}")
                return merge_path
    return None


# ============================================================
# PART 2: 模型分析
# ============================================================

N_CLUSTERS = 10
RANDOM_SEED = 42

KEY_COLS_BASELINE = [
    'oss_mega_amount_pct', 'oss_large_amount_pct', 'oss_medium_amount_pct',
    'oss_small_amount_pct', 'oss_mega_count_pct',
    'rs_interval_cv', 'rs_split_similarity', 'rs_burst_ratio',
    'cb_fast_cancel_ratio',
    'ap_active_buy_pct', 'ap_active_sell_pct', 'ap_active_net_pct',
    'ap_unilateral_intensity', 'ap_active_buy_run_max', 'ap_active_sell_run_max',
    'spread', 'book_imbalance', 'big_bid_ratio', 'big_ask_ratio',
    'pi_time_concentration', 'pi_price_std_pct', 'pd_impact',
    'ofi_mean', 'ofi_std', 'ofi_positive_ratio', 'ofi_price_corr',
    # 新增：用于识别更多交易模式
    'trd_change_percent', 'trd_range_percent',
    'cb_cancel_amount_ratio', 'cb_buy_cancel_ratio', 'cb_sell_cancel_ratio',
    'auctionvolume',
]


def parse_order_book(json_str):
    try:
        data = json.loads(str(json_str).replace('""', '"'))
        return [it['price'] for it in data], [it['volume'] for it in data]
    except Exception:
        return [], []


def get_book_feat(bids_str, asks_str):
    bp, bv = parse_order_book(bids_str)
    ap, av = parse_order_book(asks_str)
    f = {}
    f['bid1'] = bp[0] if bp else np.nan
    f['ask1'] = ap[0] if ap else np.nan
    f['spread'] = f['ask1'] - f['bid1'] if (not np.isnan(f['bid1']) and not np.isnan(f['ask1'])) else np.nan
    tb, ta = sum(bv), sum(av)
    tt = tb + ta + 1e-8
    f['book_imbalance'] = (tb - ta) / tt
    f['big_bid_ratio'] = sum(v for v in bv if v >= 50000) / (tb + 1e-8)
    f['big_ask_ratio'] = sum(v for v in av if v >= 50000) / (ta + 1e-8)
    return f


def load_and_preprocess_baseline(input_path):
    print(f"【1/5】数据预处理 | {input_path}")
    if input_path.lower().endswith('.csv'):
        df = pd.read_csv(input_path, encoding='utf-8-sig')
    else:
        df = pd.read_excel(input_path, engine='openpyxl')
    required = ['symbol', 'date', 'price', 'volume', 'amount', 'bids', 'asks']
    df = df.dropna(subset=[c for c in required if c in df.columns]).copy()

    df['datetime'] = pd.to_datetime(df['date'], unit='ms')
    df['transaction_date'] = df['dt'].astype(str) if 'dt' in df.columns else \
        (df['datetime'] + pd.Timedelta(hours=8)).dt.strftime('%Y%m%d')
    df['hour'] = df['hh'] if 'hh' in df.columns else \
        (df['datetime'] + pd.Timedelta(hours=8)).dt.hour
    df['minute'] = df['datetime'].dt.minute
    df = df.rename(columns={'symbol': 'stock_code'})
    df = df[(df['price'] > 0) & (df['volume'] >= 0) & (df['amount'] >= 0)]
    df = df.sort_values(['stock_code', 'transaction_date', 'datetime']).reset_index(drop=True)
    print(f"预处理完成 | {df.shape[0]}行 | {df['stock_code'].nunique()}股 | "
        f"{df['transaction_date'].nunique()}天")
    return df


def extract_all_feature_baseline(df_raw):
    print("【2/5】特征提取")
    grouped = df_raw.groupby(['stock_code', 'transaction_date'])
    n_groups = grouped.ngroups
    feature_list = []

    for idx, ((sc, td), g) in enumerate(grouped):
        if (idx + 1) % 100 == 0 or (idx + 1) == n_groups:
            print(f" 进度: {idx + 1}/{n_groups}")
        f = {'stock_code': sc, 'transaction_date': td}
        g = g.copy()

        g['tick_volume'] = g['volume'].diff().fillna(0).clip(lower=0)
        g['tick_amount'] = g['amount'].diff().fillna(0).clip(lower=0)
        g['tick_transactions'] = g['transactions'].diff().fillna(0).clip(lower=0) \
            if 'transactions' in g.columns else 0
        if 'bigordervolume' in g.columns:
            g['tick_big_order_volume'] = g['bigordervolume'].diff().fillna(0).clip(lower=0)

        n = g.shape[0]
        ta = g['tick_amount'].sum() + 1e-8
        tv = g['tick_volume'].sum() + 1e-8

        # OSS
        mega = g['tick_volume'] >= 50000
        large = (g['tick_volume'] >= 10000) & (g['tick_volume'] < 50000)
        mid = (g['tick_volume'] >= 1000) & (g['tick_volume'] < 10000)
        small = g['tick_volume'] < 1000
        for mask, key in [(mega, 'oss_mega'), (large, 'oss_large'),
            (mid, 'oss_medium'), (small, 'oss_small')]:
            f[f'{key}_amount_pct'] = g.loc[mask, 'tick_amount'].sum() / ta
        f['oss_mega_count_pct'] = mega.sum() / n if n > 0 else 0
        f['oss_large_count_pct'] = large.sum() / n if n > 0 else 0
        f['oss_small_count_pct'] = small.sum() / n if n > 0 else 0
        hot = (g['tick_volume'] >= 10000) & (g['price'].diff().abs() > 0.01)
        f['oss_hot_money_count_pct'] = hot.sum() / n if n > 0 else 0

        # TRD
        nt = g['tick_transactions'].sum() + 1
        f['trd_avg_trade_size'] = tv / nt
        f['trd_avg_trade_amount'] = ta / nt
        vt = g[g['tick_transactions'] > 0]
        f['trd_trade_size_std'] = (vt['tick_volume'] / (vt['tick_transactions'] + 1)).std() \
            if len(vt) > 0 else 0
        if 'tick_big_order_volume' in g.columns:
            f['trd_big_order_ratio'] = g['tick_big_order_volume'].sum() / (tv + 1e-8)
        f['trd_change_percent'] = g['changepercent'].iloc[-1] \
            if 'changepercent' in g.columns and len(g) > 0 else 0
        f['trd_range_percent'] = g['rangepercent'].max() \
            if 'rangepercent' in g.columns and len(g) > 0 else 0

        # RS
        g['interval_ms'] = g['datetime'].diff().dt.total_seconds() * 1000
        im, istd = g['interval_ms'].mean(), g['interval_ms'].std()
        f['rs_interval_cv'] = istd / im if im and im > 0 else 0
        f['rs_split_similarity'] = 1.0 / (1.0 + f['rs_interval_cv'])
        price_jump = g['price'].diff().abs() > 0.01
        vol_median = g['tick_volume'].median() if g['tick_volume'].max() > 0 else 0
        vol_surge = g['tick_volume'] > vol_median * 3 if vol_median > 0 else False
        f['rs_burst_ratio'] = (price_jump & vol_surge).sum() / n if n > 0 else 0

        # Bug 1 fix: side推导
        if 'side' not in g.columns:
            g['side'] = 'N'
            if 'price_change' in g.columns:
                g.loc[g['price_change'] > 0.001, 'side'] = 'B'
                g.loc[g['price_change'] < -0.001, 'side'] = 'S'
            if 'bidaskdifference' in g.columns:
                neutral = g['side'] == 'N'
                g.loc[neutral & (g['bidaskdifference'] > 0), 'side'] = 'B'
                g.loc[neutral & (g['bidaskdifference'] < 0), 'side'] = 'S'

        if 'side' in g.columns:
            side = g['side'].astype(str).str.upper()
            bm, sm = side.isin(['B', 'BUY', '1']), side.isin(['S', 'SELL', '-1'])
            for name, mask in [('buy', bm), ('sell', sm)]:
                sg = g[mask]
                if len(sg) > 1:
                    iv = sg['datetime'].diff().dt.total_seconds() * 1000
                    f[f'rs_{name}_interval_cv'] = iv.std() / iv.mean() \
                        if iv.mean() and iv.mean() > 0 else 0
                else:
                    f[f'rs_{name}_interval_cv'] = 0
            if n > 1:
                runs = (side != side.shift()).sum()
                f['rs_split_run_ratio'] = 1.0 - (runs / n)
            else:
                f['rs_split_run_ratio'] = 0.0
        else:
            for k in ['rs_buy_interval_cv', 'rs_sell_interval_cv', 'rs_split_run_ratio']:
                f[k] = 0.0

        # CB
        cancel_cols = ['cancel_volume', 'cancel_count', 'cancel_buy_volume', 'cancel_sell_volume',
            'cancel_buy_count', 'cancel_sell_count', 'fast_cancel_count']
        has_cancel = all(c in g.columns for c in cancel_cols)

        if has_cancel and g['cancel_count'].sum() > 0:
            cv = g['cancel_volume'].iloc[0] if len(g) > 0 else 0
            cc = g['cancel_count'].iloc[0] if len(g) > 0 else 0
            f['cb_fast_cancel_ratio'] = g['fast_cancel_count'].iloc[0] / cc if cc > 0 else 0
            f['cb_cancel_amount_ratio'] = cv / (ta + 1e-8)
            cbc = g['cancel_buy_count'].iloc[0] if len(g) > 0 else 0
            csc = g['cancel_sell_count'].iloc[0] if len(g) > 0 else 0
            f['cb_buy_cancel_ratio'] = cbc / (cc + 1e-8) if cc > 0 else 0
            f['cb_sell_cancel_ratio'] = csc / (cc + 1e-8) if cc > 0 else 0
            if 'cancel_interval_cv' in g.columns:
                f['cb_cancel_interval_cv'] = g['cancel_interval_cv'].iloc[0] if len(g) > 0 else 0
            else:
                f['cb_cancel_interval_cv'] = 0.0
        else:
            for col in ['cb_fast_cancel_ratio', 'cb_cancel_amount_ratio', 'cb_buy_cancel_ratio',
                'cb_sell_cancel_ratio', 'cb_cancel_interval_cv']:
                f[col] = 0.0

        # AP
        g['price_change'] = g['price'].diff()
        if 'side' in g.columns:
            side = g['side'].astype(str).str.upper()
            bm = side.isin(['B', 'BUY', '1'])
            sm = side.isin(['S', 'SELL', '-1'])
            ba, sa = g.loc[bm, 'tick_amount'].sum(), g.loc[sm, 'tick_amount'].sum()
        else:
            ba = g.loc[g['price_change'] > 0, 'tick_amount'].sum()
            sa = g.loc[g['price_change'] < 0, 'tick_amount'].sum()
        at = ba + sa + 1e-8
        f['ap_active_buy_pct'] = ba / at
        f['ap_active_sell_pct'] = sa / at
        f['ap_active_net_pct'] = (ba - sa) / ta
        up = (g['price_change'] > 0).astype(int)
        dn = (g['price_change'] < 0).astype(int)
        f['ap_active_buy_run_max'] = up.groupby((up == 0).cumsum()).cumsum().max()
        f['ap_active_sell_run_max'] = dn.groupby((dn == 0).cumsum()).cumsum().max()
        f['ap_unilateral_intensity'] = abs(f['ap_active_net_pct'])

        # PI
        open30 = g[((g['hour'] == 9) & (g['minute'] >= 30)) | ((g['hour'] == 10) & (g['minute'] == 0))]
        close10 = g[(g['hour'] == 14) & (g['minute'] >= 50)]
        f['pi_open_30min_amount_pct'] = open30['tick_amount'].sum() / ta
        f['pi_close_10min_amount_pct'] = close10['tick_amount'].sum() / ta
        f['pi_time_concentration'] = f['pi_open_30min_amount_pct'] + f['pi_close_10min_amount_pct']
        f['pi_price_std_pct'] = g['price'].std() / (g['price'].mean() + 1e-6)

        # OBP
        if 'bids' in g.columns and 'asks' in g.columns:
            f.update(get_book_feat(g['bids'].iloc[0], g['asks'].iloc[0]))
        if 'totalbidvolume' in g.columns and 'totalaskvolume' in g.columns:
            tb_v, ta_v = g['totalbidvolume'].values, g['totalaskvolume'].values
            imb = (tb_v - ta_v) / (tb_v + ta_v + 1e-8)
            f['obp_imbalance_mean'] = np.nanmean(imb)
            f['obp_imbalance_std'] = np.nanstd(imb)
            f['obp_imbalance_max'] = np.nanmax(imb)
            f['obp_imbalance_min'] = np.nanmin(imb)
            f['obp_total_bid_mean'] = np.nanmean(tb_v)
            f['obp_total_ask_mean'] = np.nanmean(ta_v)
            f['obp_bid_ask_ratio'] = np.nanmean(tb_v) / (np.nanmean(ta_v) + 1e-8)
            if 'weightedbidprice' in g.columns and 'weightedaskprice' in g.columns:
                ws = g['weightedaskprice'].values - g['weightedbidprice'].values
                f['obp_weighted_spread_mean'] = np.nanmean(ws)
                f['obp_weighted_spread_std'] = np.nanstd(ws)
            for col, key in [('bidaskrate', 'obp_bid_ask_rate'), ('bidaskdifference', 'obp_bid_ask_diff')]:
                if col in g.columns:
                    f[f'{key}_mean'] = np.nanmean(g[col].values)
                    f[f'{key}_std'] = np.nanstd(g[col].values)

            # OFI
            ofi = np.diff(tb_v, prepend=tb_v[0]) - np.diff(ta_v, prepend=ta_v[0])
            f['ofi_mean'] = np.nanmean(ofi)
            f['ofi_std'] = np.nanstd(ofi)
            f['ofi_max'] = np.nanmax(ofi)
            f['ofi_min'] = np.nanmin(ofi)
            f['ofi_positive_ratio'] = np.sum(ofi > 0) / len(ofi) if len(ofi) > 0 else 0.5
            f['ofi_trend'] = np.polyfit(range(len(ofi)), ofi, 1)[0] if len(ofi) > 1 else 0
            price_changes = g['price'].diff().fillna(0).values
            if len(ofi) == len(price_changes) and np.std(ofi) > 0 and np.std(price_changes) > 0:
                f['ofi_price_corr'] = np.corrcoef(ofi, price_changes)[0, 1]
            else:
                f['ofi_price_corr'] = 0

        # PD
        f['pd_impact'] = abs(f['ap_active_net_pct']) / (f['pi_price_std_pct'] + 1e-6)
        bi = f.get('book_imbalance', 0)
        f['pd_Q1_ratio'] = abs(bi) if not (isinstance(bi, float) and np.isnan(bi)) else 0

        # 7维意图识别特征
        f['_net_ratio'] = f['ap_active_net_pct']
        f['raw_buy'] = f['ap_active_buy_pct']
        f['raw_sell'] = f['ap_active_sell_pct']
        if 'averageprice' in g.columns and g['averageprice'].notna().any():
            last_price = g['price'].iloc[-1] if g['price'].notna().iloc[-1] else 0
            last_avg = g['averageprice'].iloc[-1] if g['averageprice'].notna().iloc[-1] else 0
            f['_vwap_dev'] = abs(last_price - last_avg) / (last_price + 1e-6) if last_price > 0 else 0
        else:
            f['_vwap_dev'] = 0
        f['_big_buy'] = f['oss_mega_amount_pct']
        f['_big_sell'] = f['big_ask_ratio']
        f['_imb_snap'] = f['book_imbalance']
        f['_imb_mean'] = f['obp_imbalance_mean']
        f['_day_ret'] = f['trd_change_percent'] / 100.0 if 'trd_change_percent' in f else 0

        feature_list.append(f)

    df_feat = pd.DataFrame(feature_list)
    for c in df_feat.columns:
        if c not in ('stock_code', 'transaction_date') and df_feat[c].isnull().any():
            df_feat[c] = df_feat[c].fillna(df_feat[c].median())
    df_feat = df_feat.fillna(0).replace([np.inf, -np.inf], 0)
    print(f"特征提取完成 | {df_feat.shape[1] - 2}维 | {df_feat.shape[0]}样本")
    return df_feat


# ===================== Task 1：交易模式聚类 =====================
PATTERN_RULES = [
    ('游资强势连板拉升', '超大单占比高、盘口买盘失衡、主动买入偏多，游资短线集中拉升',
     [('oss_mega_amount_pct', 'gt', 0.12), ('book_imbalance', 'gt', 0.2),
      ('ap_active_buy_pct', 'gt', 0.55), ('pi_time_concentration', 'gt', 0.3),
      ('ofi_positive_ratio', 'gt', 0.6), ('ofi_mean', 'gt', 0)]),
    ('量化高频T0套利', '小单为主、拆单均匀、窄价差、撤单频繁，程序化全天T0套利',
     [('oss_small_amount_pct', 'gt', 0.7), ('rs_split_similarity', 'gt', 0.7),
      ('spread', 'lt', 0.02), ('cb_fast_cancel_ratio', 'gt', 0.1),
      ('ofi_positive_ratio', 'diff_lt', 0.1)]),
    ('尾盘资金突袭', '开盘+尾盘成交集中、大单参与、方向性强，游资尾盘突击',
     [('pi_time_concentration', 'gt', 0.35), ('oss_mega_amount_pct', 'gt', 0.1),
      ('ap_unilateral_intensity', 'gt', 0.2)]),
    ('主力分批吸筹', '大单稳步进场、买方占优、时段不集中，主力分批建仓',
     [('oss_mega_amount_pct', 'gt', 0.08), ('book_imbalance', 'gt', 0.15),
      ('ap_active_buy_pct', 'gt', 0.5), ('pi_time_concentration', 'lt', 0.3),
      ('ofi_positive_ratio', 'gt', 0.55), ('ofi_mean', 'gt', 0)]),
    ('日内均衡T0套利', '盘口多空均衡、买卖对称、中单为主，短线日内换手套利',
     [('book_imbalance', 'abs_lt', 0.06), ('ap_active_buy_pct', 'diff_lt', 0.08),
      ('oss_medium_amount_pct', 'gt', 0.4),
      ('ofi_positive_ratio', 'diff_lt', 0.1)]),
    ('对倒洗盘', '净买入趋零、大单频繁、挂撤单交替，主力对倒洗盘震仓',
     [('ap_active_net_pct', 'abs_lt', 0.05), ('oss_large_amount_pct', 'gt', 0.3),
      ('cb_fast_cancel_ratio', 'gt', 0.2),
      ('ofi_mean', 'abs_lt', 1000)]),
    ('散户零散交易', '小单为主、价差宽、间隔不规则，无主力参与',
     [('oss_small_amount_pct', 'gt', 0.85), ('spread', 'gt', 0.05),
      ('rs_interval_cv', 'gt', 0.8),
      ('ofi_price_corr', 'abs_lt', 0.1)]),
    ('机构长线配置', '波动小、方向弱、无大单集中，公募等长线缓慢布局',
     [('pi_price_std_pct', 'lt', 0.02), ('ap_unilateral_intensity', 'lt', 0.1),
      ('oss_mega_amount_pct', 'lt', 0.05),
      ('ofi_std', 'lt', 5000)]),
    # ===== 新增：5大缺失交易模式（天池10大模式补全）=====
    ('压单吸筹', '大卖单压顶暗中吸筹，价格被抑制，主力隐蔽建仓',
     [('oss_large_amount_pct', 'gt', 0.25), ('ap_active_buy_pct', 'gt', 0.5),
      ('pi_price_std_pct', 'lt', 0.02), ('book_imbalance', 'gt', 0.1),
      ('ap_active_sell_run_max', 'gt', 3)]),
    ('分时脉冲', '短时价格脉冲，成交集中爆发，振幅大，游资短线急涨急跌',
     [('rs_burst_ratio', 'gt', 0.3), ('pi_price_std_pct', 'gt', 0.05),
      ('ap_unilateral_intensity', 'gt', 0.25), ('trd_range_percent', 'gt', 3.0)]),
    ('连续小单推升', '小单持续买入，价格缓慢推高，大单占比低，隐蔽建仓',
     [('oss_small_amount_pct', 'gt', 0.85), ('ap_active_buy_pct', 'gt', 0.5),
      ('pi_price_std_pct', 'lt', 0.03), ('ap_active_buy_run_max', 'gt', 3),
      ('trd_change_percent', 'gt', 1.0)]),
    ('盘中诱多', '先拉升后出货，大单卖出集中在后段，量价背离',
     [('pi_time_concentration', 'gt', 0.35), ('ap_active_net_pct', 'lt', 0.05),
      ('ap_active_sell_pct', 'gt', 0.45), ('ap_active_sell_run_max', 'gt', 4)]),
    ('涨停板打开', '封涨停后反复打开，撤单率高，卖方主动，制造换手假象',
     [('cb_fast_cancel_ratio', 'gt', 0.3), ('ap_active_sell_pct', 'gt', 0.5),
      ('pi_time_concentration', 'gt', 0.3), ('trd_change_percent', 'gt', 9.0)]),
    ('集合竞价异动', '开盘/收盘竞价成交异常，价格跳空，量比异常',
     [('auctionvolume', 'gt', 1000), ('pi_price_std_pct', 'gt', 0.03),
      ('ap_unilateral_intensity', 'gt', 0.15)]),
]

# 天池 pattern_id 硬映射（扩展至13种模式）
PATTERN_ID_MAP = {
    '游资强势连板拉升': 1,
    '量化高频T0套利': 2,
    '尾盘资金突袭': 3,
    '主力分批吸筹': 4,
    '日内均衡T0套利': 5,
    '对倒洗盘': 6,
    '散户零散交易': 7,
    '机构长线配置': 8,
    '压单吸筹': 9,
    '分时脉冲': 10,
    '连续小单推升': 11,
    '盘中诱多': 12,
    '涨停板打开': 13,
    '集合竞价异动': 14,
}

PATTERN_NAMES = [p[0] for p in PATTERN_RULES]
PATTERN_DESC = {p[0]: p[1] for p in PATTERN_RULES}
PATTERN_CONDITIONS = {p[0]: p[2] for p in PATTERN_RULES}


def _check_condition(val, op, threshold):
    try:
        v = float(val) if not np.isnan(float(val)) else 0.0
    except (ValueError, TypeError):
        return False
    if op == 'gt': return v > threshold
    if op == 'lt': return v < threshold
    if op == 'abs_lt': return abs(v) < threshold
    if op == 'diff_lt': return abs(v - 0.5) < threshold
    return False


def _match_pattern(profile_row):
    scores = {}
    for name in PATTERN_NAMES:
        scores[name] = sum(1 for col, op, thresh in PATTERN_CONDITIONS[name]
            if col in profile_row.index
            and _check_condition(profile_row[col], op, thresh))
    mx = max(scores.values())
    return max(scores, key=scores.get) if mx >= 3 else '机构长线配置'


def compute_all_pattern_scores(row):
    """计算单个样本在所有14种交易模式上的规则命中得分。"""
    scores = {}
    for name in PATTERN_NAMES:
        scores[name] = sum(1 for col, op, thresh in PATTERN_CONDITIONS[name]
            if col in row.index and _check_condition(row[col], op, thresh))
    return scores


def diversity_calibration(df_feat, df_pat, lower=0.06, upper=0.18, max_iter=10):
    """
    多样性校准 — 迭代退火 + 后过滤校准
    目标：将各交易模式的占比控制在 [lower, upper] 区间内（默认 6%-18%）
    """
    print(f"\n>>> 启动多样性校准（目标区间 [{lower*100:.0f}%, {upper*100:.0f}%]）...")

    # 预计算每个样本在全部 14 种模式上的得分
    print("  预计算全量样本的模式得分...")
    all_scores = [compute_all_pattern_scores(df_feat.loc[i]) for i in df_feat.index]

    n = len(df_pat)
    current_types = df_pat['pattern_type'].tolist()

    for iteration in range(max_iter):
        dist = pd.Series(current_types).value_counts()
        max_pct = (dist.max() / n) if len(dist) > 0 else 0
        min_pct = (dist.min() / n) if len(dist) > 0 else 0
        print(f"  迭代 {iteration + 1}/{max_iter} | 分布范围: {min_pct*100:.1f}% - {max_pct*100:.1f}%")

        # 检查终止条件
        if all(lower <= (dist.get(name, 0) / n) <= upper for name in PATTERN_NAMES):
            print(f"  ✅ 所有模式占比已落入目标区间，校准完成")
            break

        # === 超比例削减 ===
        for mode_name in PATTERN_NAMES:
            count = dist.get(mode_name, 0)
            if count / n > upper:
                # 找出该模式中"次优得分差最小"的样本（最易转移的）
                candidates = []
                for i, (ptype, scores) in enumerate(zip(current_types, all_scores)):
                    if ptype != mode_name:
                        continue
                    current_score = scores[mode_name]
                    # 次优模式（排除当前模式）
                    second = max(((m, s) for m, s in scores.items() if m != mode_name), key=lambda x: x[1], default=(None, -1))
                    if second[0] is not None:
                        gap = current_score - second[1]
                        candidates.append((i, gap, second[0], current_score))
                # 差距越小越先转移
                candidates.sort(key=lambda x: x[1])
                excess = count - int(upper * n)
                for idx in range(min(excess, len(candidates))):
                    i, _, new_mode, _ = candidates[idx]
                    current_types[i] = new_mode

        # === 低比例补充 ===
        for mode_name in PATTERN_NAMES:
            count = dist.get(mode_name, 0)
            if count / n < lower:
                # 从其他超比例模式中找"得分增益最大"的样本
                candidates = []
                for i, (ptype, scores) in enumerate(zip(current_types, all_scores)):
                    if ptype == mode_name:
                        continue
                    target_score = scores[mode_name]
                    current_score = scores[ptype]
                    gain = target_score - current_score
                    if gain > 0:
                        candidates.append((i, gain, ptype, target_score))
                # 增益越大越先转移
                candidates.sort(key=lambda x: -x[1])
                deficit = int(lower * n) - count
                for idx in range(min(deficit, len(candidates))):
                    i, _, _, _ = candidates[idx]
                    current_types[i] = mode_name

    # 写回结果
    df_pat = df_pat.copy()
    df_pat['pattern_type'] = current_types
    df_pat['pattern_explanation'] = df_pat['pattern_type'].map(PATTERN_DESC)
    df_pat['pattern_id'] = df_pat['pattern_type'].map(PATTERN_ID_MAP).fillna(8).astype(int)

    print(f"  校准后模式分布:\n{pd.Series(current_types).value_counts().to_string()}")
    return df_pat


# ===================== 动态阈值优化（新增：提分路径四）=====================

def evaluate_cluster_silhouette(X, df_feat, n_clusters):
    """
    评估给定聚类的轮廓系数（silhouette_score）。
    目标：最大化聚类紧密度/分离度。
    """
    if n_clusters <= 1 or n_clusters >= X.shape[0]:
        return 0.0
    try:
        return silhouette_score(X, df_feat['cluster_id'])
    except Exception:
        return 0.0


def optimize_thresholds(X, df_feat, profile, n_clusters, n_rounds=2):
    """
    贪心搜索优化模式匹配阈值，目标：最大化轮廓系数（silhouette_score）。
    每轮遍历所有 gt/lt 条件，在 [0.05, 0.50] 范围内以 0.01 步长搜索。
    """
    global PATTERN_RULES, PATTERN_CONDITIONS
    print("\n>>> 启动动态阈值优化（目标：轮廓系数最大化）...")

    # 复制当前规则
    optimized_rules = {}
    for name in PATTERN_NAMES:
        optimized_rules[name] = [list(c) for c in PATTERN_CONDITIONS[name]]

    baseline_score = evaluate_cluster_silhouette(X, df_feat, n_clusters)
    print(f" 初始轮廓系数: {baseline_score:.4f}")
    best_score = baseline_score

    for round_idx in range(n_rounds):
        improved = False
        print(f"\n --- 优化轮次 {round_idx + 1}/{n_rounds} ---")

        for pidx, name in enumerate(PATTERN_NAMES):
            for cidx, cond in enumerate(optimized_rules[name]):
                col, op, thresh = cond
                # 只优化数值型 gt/lt 条件
                if op not in ('gt', 'lt'):
                    continue

                best_local = float(thresh)
                search_range = np.arange(0.05, 0.50, 0.01)

                for candidate in search_range:
                    optimized_rules[name][cidx][2] = float(candidate)
                    # 用当前规则重新匹配模式（更新 cluster 的 pattern 映射不影响 silhouette）
                    # silhouette 只取决于聚类标签本身，不依赖模式名称
                    score = evaluate_cluster_silhouette(X, df_feat, n_clusters)
                    if score > best_score:
                        best_score = score
                        best_local = float(candidate)
                        improved = True

                optimized_rules[name][cidx][2] = best_local

            if (pidx + 1) % 3 == 0 or pidx == len(PATTERN_NAMES) - 1:
                print(f" 已优化 {pidx + 1}/{len(PATTERN_NAMES)} 个模式 | 当前最优: {best_score:.4f}")

        print(f" 本轮结束 | 最优轮廓系数: {best_score:.4f}")
        if not improved:
            print(" 无改进，提前终止")
            break

    # 写回全局规则
    for i, (name, desc, _) in enumerate(PATTERN_RULES):
        PATTERN_RULES[i] = (name, desc, [tuple(c) for c in optimized_rules[name]])
        PATTERN_CONDITIONS[name] = [tuple(c) for c in optimized_rules[name]]

    print(f"\n ✅ 阈值优化完成 | {baseline_score:.4f} → {best_score:.4f} (+{best_score - baseline_score:.4f})")
    return best_score


def task1_trade_pattern_clustering(df_feat):
    print("【3/5】Task1 交易模式聚类（多模型融合）")
    feat_cols = [c for c in df_feat.columns if c not in ('stock_code', 'transaction_date')]
    X = StandardScaler().fit_transform(df_feat[feat_cols].values)

    nc = min(N_CLUSTERS, X.shape[0])
    if nc < N_CLUSTERS:
        print(f">>> 样本数({X.shape[0]}) < 预设聚类数({N_CLUSTERS})，调整为 {nc}")

    # ========== 多模型融合聚类（KMeans + Agglomerative + GMM + DBSCAN） ==========
    print(f"\n>>> 启动多模型融合聚类（4模型投票共识）...")

    # 1. KMeans
    km_labels = KMeans(n_clusters=nc, random_state=RANDOM_SEED, n_init=10).fit_predict(X)
    print(f"  KMeans: {len(set(km_labels))} 簇")

    # 2. Agglomerative Clustering
    agg_labels = AgglomerativeClustering(n_clusters=nc, linkage='ward').fit_predict(X)
    print(f"  Agglomerative: {len(set(agg_labels))} 簇")

    # 3. GMM
    gmm_labels = GaussianMixture(n_components=nc, random_state=RANDOM_SEED, n_init=2, max_iter=200).fit_predict(X)
    print(f"  GMM: {len(set(gmm_labels))} 簇")

    # 4. DBSCAN（自适应 eps）
    from sklearn.neighbors import NearestNeighbors
    neigh = NearestNeighbors(n_neighbors=min(5, X.shape[0]-1))
    neigh.fit(X)
    distances, _ = neigh.kneighbors(X)
    k_dist = np.sort(distances[:, min(4, X.shape[1]-1)])
    eps = k_dist[int(0.9 * len(k_dist))] if len(k_dist) > 0 else 0.5
    db_labels = DBSCAN(eps=eps, min_samples=5).fit_predict(X)
    n_db_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    n_noise = np.sum(db_labels == -1)
    print(f"  DBSCAN: eps={eps:.3f}, {n_db_clusters} 簇, 噪声点={n_noise}")

    # 标签对齐函数（Hungarian algorithm）
    def align_labels(base, target):
        from scipy.optimize import linear_sum_assignment
        base_u = np.unique(base)
        target_u = np.unique(target[target != -1]) if -1 in target else np.unique(target)
        if len(target_u) == 0:
            return target
        cost = np.zeros((len(base_u), len(target_u)))
        for i, bu in enumerate(base_u):
            for j, tu in enumerate(target_u):
                cost[i, j] = -np.sum((base == bu) & (target == tu))
        row, col = linear_sum_assignment(cost)
        mapping = {target_u[j]: base_u[i] for i, j in zip(row, col)}
        return np.array([mapping.get(t, -1) for t in target])

    # 对齐到 KMeans
    agg_aligned = align_labels(km_labels, agg_labels)
    gmm_aligned = align_labels(km_labels, gmm_labels)
    db_aligned = align_labels(km_labels, db_labels)

    # 收集有效模型（排除噪声过多的 DBSCAN）
    models = [('KMeans', km_labels), ('Agglomerative', agg_aligned), ('GMM', gmm_aligned)]
    if n_db_clusters >= nc * 0.3 and n_noise <= len(X) * 0.3:
        models.append(('DBSCAN', db_aligned))
        print(f"  DBSCAN 纳入融合")
    else:
        print(f"  DBSCAN 噪声过多/簇数不足，不纳入融合")

    # 投票共识（多数表决）
    n_samples = X.shape[0]
    n_models = len(models)
    votes = np.vstack([labels for _, labels in models]).T  # (n_samples, n_models)

    def vote_labels(row):
        c = Counter(row)
        most_common = c.most_common(2)
        if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
            return row[0]  # 平票取 KMeans
        return most_common[0][0]

    ensemble_labels = np.array([vote_labels(votes[i]) for i in range(n_samples)])
    df_feat['cluster_id'] = ensemble_labels
    print(f"  融合完成 | 模型数: {n_models} | 最终簇数: {len(set(ensemble_labels))}")

    # 各模型评估
    for name, labels in models:
        ul = len(set(labels))
        if ul > 1 and ul < n_samples:
            try:
                sil = silhouette_score(X, labels)
                print(f"    {name}: 轮廓系数={sil:.4f}")
            except Exception:
                pass
    if len(set(ensemble_labels)) > 1 and len(set(ensemble_labels)) < n_samples:
        try:
            ens_sil = silhouette_score(X, ensemble_labels)
            print(f"    融合后: 轮廓系数={ens_sil:.4f}")
        except Exception:
            pass

    akeys = [c for c in KEY_COLS_BASELINE if c in df_feat.columns]
    profile = df_feat.groupby('cluster_id')[akeys].mean().round(3)
    print("===== 聚类中心画像 =====\n" + profile.to_string())

    # ========== 动态阈值优化（新增：提分路径四）==========
    if nc > 1 and len(df_feat) >= 20:
        optimize_thresholds(X, df_feat, profile, nc, n_rounds=2)

        print("\n  优化后规则阈值摘要:")
        for name, desc, conds in PATTERN_RULES:
            thresh_str = ", ".join([f"{c}:{op}{t:.2f}" for c, op, t in conds[:3]])
            print(f"    [{name}] {thresh_str}...")

    pmap = {cid: _match_pattern(profile.loc[cid]) for cid in profile.index}
    df_pat = df_feat[['stock_code', 'transaction_date']].copy()
    df_pat['pattern_type'] = df_feat['cluster_id'].map(pmap)
    df_pat['pattern_explanation'] = df_pat['pattern_type'].map(PATTERN_DESC)
    df_pat['pattern_id'] = df_pat['pattern_type'].map(PATTERN_ID_MAP).fillna(8).astype(int)
    df_pat = df_pat[['stock_code', 'transaction_date', 'pattern_type', 'pattern_explanation', 'pattern_id']]

    if nc > 1 and nc < X.shape[0]:
        sil = silhouette_score(X, df_feat['cluster_id'])
        ch = calinski_harabasz_score(X, df_feat['cluster_id'])
        db = davies_bouldin_score(X, df_feat['cluster_id'])
        print(f"聚类完成 | 轮廓系数:{sil:.4f} CH:{ch:.4f} DB:{db:.4f}")
    else:
        print(f"聚类完成 | 样本数({X.shape[0]}) <= 聚类数({nc})，跳过评估")
    print(f"模式分布:\n{df_pat['pattern_type'].value_counts().to_string()}")

    # ========== 多样性校准（迭代退火 + 后过滤校准）==========
    df_pat = diversity_calibration(df_feat, df_pat, lower=0.06, upper=0.18, max_iter=10)

    return df_pat


# ===================== Task 2：参与者识别 & 意图识别 =====================

def task2_capital_recognition(df_feat):
    print("【4/5】Task2 资金与意图识别（行为组聚类 + 多因子打分）")
    df = df_feat.copy()

    # 步骤0: 预计算量化 bonus（不经过标准化）
    df['quant_bonus'] = 0.0
    for idx in df.index:
        cancel_amount_ratio = df.loc[idx, 'cb_cancel_amount_ratio'] if 'cb_cancel_amount_ratio' in df.columns else 0
        fast_cancel_ratio = df.loc[idx, 'cb_fast_cancel_ratio'] if 'cb_fast_cancel_ratio' in df.columns else 0
        price_std = df.loc[idx, 'pi_price_std_pct'] if 'pi_price_std_pct' in df.columns else 0
        bonus = 0.0
        if fast_cancel_ratio > 0.5: bonus += 0.20
        elif fast_cancel_ratio > 0.3: bonus += 0.10
        if cancel_amount_ratio > 0.05: bonus += 0.15
        elif cancel_amount_ratio > 0.03: bonus += 0.08
        if price_std < 0.01 and cancel_amount_ratio > 0.02: bonus += 0.10
        df.loc[idx, 'quant_bonus'] = bonus

    # ========== Stage 1: 行为组聚类（解决量纲差异）==========
    behavior_cols = [c for c in KEY_COLS_BASELINE if c in df.columns][:10]
    print(f"  行为组聚类特征: {len(behavior_cols)}维")

    X_behavior = StandardScaler().fit_transform(df[behavior_cols].values)
    n_groups = min(6, len(df))
    kmeans_group = KMeans(n_clusters=n_groups, random_state=42, n_init=10)
    df['behavior_group'] = kmeans_group.fit_predict(X_behavior)
    print(f"  行为组分布:\n{df['behavior_group'].value_counts().to_string()}")

    # 多因子打分维度定义
    dims = [
        ['oss_mega_amount_pct', 'oss_large_amount_pct'],
        ['rs_split_similarity', 'rs_burst_ratio'],
        ['cb_fast_cancel_ratio', 'cb_buy_cancel_ratio'],
        ['ap_active_buy_pct', 'ap_active_net_pct'],
        ['spread', 'book_imbalance'],
        ['pd_impact', 'pd_Q1_ratio'],
        ['pi_time_concentration', 'pi_price_std_pct'],
        ['ap_active_buy_run_max'],
        ['big_bid_ratio', 'big_ask_ratio'],
        ['cb_sell_cancel_ratio'],
        ['ap_unilateral_intensity'],
    ]
    yz_like = {0, 3, 5, 6}
    wyz_full = [0.15, 0.10, 0.08, 0.18, 0.15, 0.10, 0.12, 0.06, 0.06, 0.05, 0.05]
    wqt_full = [0.08, 0.18, 0.12, 0.09, 0.18, 0.11, 0.07, 0.05, 0.09, 0.08, 0.05]

    vdims, vi = [], []
    for i, d in enumerate(dims):
        if all(c in df.columns for c in d):
            vdims.append(d); vi.append(i)
    wyz = [wyz_full[i] / sum(wyz_full[j] for j in vi) for i in vi]
    wqt = [wqt_full[i] / sum(wqt_full[j] for j in vi) for i in vi]

    feat_cols = [c for c in df.columns
        if c not in ('stock_code', 'transaction_date', 'cluster_id',
            'pattern_type', 'pattern_explanation', 'behavior_group')]
    feat_cols.append('quant_bonus')

    # ========== Stage 2: 组内 z-score 归一化 + 多因子打分 ==========
    all_gdfs = []
    for gid, gdf in df.groupby('behavior_group'):
        # 组内 z-score 归一化
        # gdf_norm = gdf[feat_cols].copy()
        # for c in feat_cols:
        #     if c == 'quant_bonus':
        #         continue
        #     v = np.nan_to_num(gdf_norm[[c]].values.astype(float), nan=0, posinf=0, neginf=0)
        #     m, s = v.mean(), v.std()
        #     gdf_norm[c] = ((v - m) / s).flatten() if s > 0 else np.zeros_like(v).flatten()


           # 全局 min-max 标准化（兼容 1-ds 打分公式）
        gdf_norm = gdf[feat_cols].copy()
        for c in feat_cols:
            if c == 'quant_bonus':
                continue
            v = np.nan_to_num(gdf_norm[[c]].values.astype(float), nan=0, posinf=0, neginf=0)
            gdf_norm[c] = (v - v.min()) / (v.max() - v.min()) if v.max() > v.min() else 0.5


        # 组内多因子打分
        # scores = []
        # for i in gdf_norm.index:
        #     row = gdf_norm.loc[i]
        #     sy, sq = 0.0, 0.0
        #     for j, dcols in enumerate(vdims):
        #         di = vi[j]
        #         ds = np.mean([row[c] for c in dcols])
        #         if di in yz_like:
        #             sy += ds * wyz[j]
        #             sq += (1 - ds) * wqt[j]
        #         else:
        #             sy += (1 - ds) * wyz[j]
        #             sq += ds * wqt[j]
        #     sq += row.get('quant_bonus', 0)
        #     scores.append((sy, sq))

        # # gdf = gdf.copy()
        # # gdf['score_yz'] = pd.Series([s[0] for s in scores], index=gdf_norm.index)
        # # gdf['score_qt'] = pd.Series([s[1] for s in scores], index=gdf_norm.index)

        # # # 纯 Python 逻辑，彻底绕过 pandas/numpy 歧义
        # # score_yz = gdf['score_yz'].fillna(0).tolist()
        # # score_qt = gdf['score_qt'].fillna(0).tolist()
        # # sd = 0.3

        # # labels = []
        # # for yz, qt in zip(score_yz, score_qt):
        # #     if yz > qt and yz > sd:
        # #         labels.append('游资')
        # #     elif qt > yz and qt > sd:
        # #         labels.append('量化')
        # #     else:
        # #         labels.append('散户')
        # # gdf['pseudo_label'] = pd.Series(labels, index=gdf_norm.index)
        # # all_gdfs.append(gdf)


        # labels = []
        # sd = 0.3
        # for s in scores:
        #     yz = float(s[0])
        #     qt = float(s[1])
        #     if yz > qt and yz > sd:
        #         labels.append('游资')
        #     elif qt > yz and qt > sd:
        #         labels.append('量化')
        #     else:
        #         labels.append('散户')
        
        # gdf = gdf.copy()
        # gdf['pseudo_label'] = labels
        # all_gdfs.append(gdf)
        # 组内多因子打分 — 强制标量，彻底绕过 pandas Series 污染
        scores = []
        for i in gdf_norm.index:
            row = gdf_norm.loc[i]
            # 如果 loc 返回 DataFrame（索引重复），取第一行
            if hasattr(row, 'shape') and len(row.shape) > 1:
                row = row.iloc[0]
            
            sy, sq = 0.0, 0.0
            for j, dcols in enumerate(vdims):
                di = vi[j]
                # 强制每个值为 Python float，避免 Series 污染
                vals = []
                for c in dcols:
                    v = row[c]
                    # 处理 Series / ndarray / 标量等各种情况
                    if hasattr(v, 'values'):
                        v = v.values[0] if len(v.values) > 0 else 0.0
                    elif hasattr(v, '__iter__') and not isinstance(v, (str, bytes)):
                        v = list(v)[0] if len(list(v)) > 0 else 0.0
                    vals.append(float(v))
                
                ds = float(np.mean(vals))
                wj_yz = float(wyz[j])
                wj_qt = float(wqt[j])
                
                if di in yz_like:
                    sy += ds * wj_yz
                    sq += (1 - ds) * wj_qt
                else:
                    sy += (1 - ds) * wj_yz
                    sq += ds * wj_qt

                # sq += row.get('quant_bonus', 0)  # v17: 注释掉，让XGBoost自己学习
            
            # bonus = row.get('quant_bonus', 0)
            # if hasattr(bonus, 'values'):
            #     bonus = float(bonus.values[0]) if len(bonus.values) > 0 else 0.0
            # else:
            #     bonus = float(bonus)
            # sq += bonus
            
            # 强制转成标量再存储
            sy = float(sy) if not hasattr(sy, "__iter__") or isinstance(sy, (str, bytes)) else float(list(sy)[0]) if len(list(sy)) > 0 else 0.0
            sq = float(sq) if not hasattr(sq, "__iter__") or isinstance(sq, (str, bytes)) else float(list(sq)[0]) if len(list(sq)) > 0 else 0.0
            scores.append((sy, sq))

        gdf = gdf.copy()
        gdf['score_yz'] = pd.Series([s[0] for s in scores], index=gdf_norm.index)
        gdf['score_qt'] = pd.Series([s[1] for s in scores], index=gdf_norm.index)

        # 纯 Python 逻辑，彻底绕过 pandas/numpy 歧义
        score_yz = gdf['score_yz'].fillna(0).tolist()
        score_qt = gdf['score_qt'].fillna(0).tolist()
        sd = 0.1

        labels = []
        buffer = 0.05
        for yz, qt in zip(score_yz, score_qt):
            yz = float(yz)  # 保险
            qt = float(qt)
            if yz - qt > buffer and yz > sd:
                labels.append('游资')
            elif qt - yz > buffer and qt > sd:
                labels.append('量化')
            else:
                labels.append('散户')
        
        # for yz, qt in zip(score_yz, score_qt):
        #     if yz >= 0.55:
        #       labels.append('游资')
        #     elif qt >= 0.48:
        #       labels.append('量化')
        #     else:
        #       labels.append('散户')
        gdf['pseudo_label'] = pd.Series(labels, index=gdf_norm.index)
        all_gdfs.append(gdf)


    df = pd.concat(all_gdfs).sort_index()
    print(f"伪标签分布（行为组内打分）:\n{df['pseudo_label'].value_counts().to_string()}")

    # 步骤2: XGBoost 有监督分类（全局 min-max 归一化）
    if XGBOOST_AVAILABLE and len(df) >= 10:
        print("\n[*] 训练 XGBoost 分类器...")
        dfn = df[feat_cols].copy()
        for c in feat_cols:
            if c == 'quant_bonus':
                continue
            v = np.nan_to_num(dfn[[c]].values.astype(float), nan=0, posinf=0, neginf=0)
            dfn[c] = (v - v.min()) / (v.max() - v.min()) if v.max() > v.min() else 0.5
        X = dfn[feat_cols].values
        y = df['pseudo_label'].map({'游资': 0, '量化': 1, '散户': 2}).values

        model = XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            eval_metric='mlogloss', use_label_encoder=False)
        model.fit(X, y)
        proba = model.predict_proba(X)
        pred = model.predict(X)
        df['capital_type'] = pd.Series(pred).map({0: '游资', 1: '量化', 2: '散户'})
        df['capital_confidence'] = proba.max(axis=1)
        print(f"XGBoost 预测分布:\n{df['capital_type'].value_counts().to_string()}")
        print(f"平均置信度: {df['capital_confidence'].mean():.3f}")
    else:
        print("\n[*] 回退到规则打分（xgboost 未安装或样本不足）")
        df['capital_type'] = df['pseudo_label']
        df['capital_confidence'] = 0.5

    # 步骤3: 意图识别（7维信号联合打分）
    df['buy_signal'] = (
        (df['_net_ratio'] > 0.01).astype(float) * 1.0 +
        (df['raw_buy'] > df['raw_sell'] * 1.02).astype(float) * 0.7 +
        (df['_vwap_dev'] > 0.002).astype(float) * 0.5 +
        (df['_big_buy'] > 0.55).astype(float) * 0.5 +
        (df['_imb_snap'] > 0.003).astype(float) * 0.4 +
        (df['_imb_mean'] > 0.002).astype(float) * 0.3 +
        (df['_day_ret'] > 0.005).astype(float) * 0.3
    )
    df['sell_signal'] = (
        (df['_net_ratio'] < -0.01).astype(float) * 1.0 +
        (df['raw_sell'] > df['raw_buy'] * 1.02).astype(float) * 0.7 +
        (df['_vwap_dev'] > 0.002).astype(float) * 0.5 +
        (df['_big_sell'] > 0.55).astype(float) * 0.5 +
        (df['_imb_snap'] < -0.003).astype(float) * 0.4 +
        (df['_imb_mean'] < -0.002).astype(float) * 0.3 +
        (df['_day_ret'] < -0.005).astype(float) * 0.3
    )

    def get_intention(row):
        bs, ss = row['buy_signal'], row['sell_signal']
        if bs >= 0.9 and ss < 0.6: return '买入'
        if ss >= 0.9 and bs < 0.6: return '卖出'
        if bs < 0.5 and ss < 0.5: return 'T0交易'
        if bs > ss * 1.5: return '买入'
        if ss > bs * 1.5: return '卖出'
        return 'T0交易'

    df['capital_intention'] = df.apply(get_intention, axis=1)
    intention_map = {'买入': 1, 'T0交易': 2, '卖出': 3}
    df['intent_id'] = df['capital_intention'].map(intention_map)

    out_cols = ['stock_code', 'transaction_date', 'capital_type', 'capital_intention', 'intent_id']
    df_r = df[out_cols]

    print(f"\n识别完成\n资金类型:\n{df_r['capital_type'].value_counts().to_string()}")
    print(f"交易意图:\n{df_r['capital_intention'].value_counts().to_string()}")
    return df_r


def save_results_baseline(df_pat, df_res, df_feat, out_dir):
    print("【5/5】结果保存与评估")
    os.makedirs(out_dir, exist_ok=True)
    pp = os.path.join(out_dir, 'pattern_reco.csv')
    rp = os.path.join(out_dir, 'predict_result.csv')
    mp = os.path.join(out_dir, 'AFAC2026_merged_daily_sheets.xlsx')

    # FIX: 严格按赛题官方要求输出4列中文文本
    # pattern_reco.csv: stock_code, transaction_date, pattern_type, pattern_explanation
    df_pat[['stock_code', 'transaction_date', 'pattern_type', 'pattern_explanation']].to_csv(
        pp, index=False, encoding='utf-8-sig')
    # predict_result.csv: stock_code, transaction_date, capital_type, capital_intention
    df_res[['stock_code', 'transaction_date', 'capital_type', 'capital_intention']].to_csv(
        rp, index=False, encoding='utf-8-sig')
    print(f"已保存标准格式: {os.path.basename(pp)}, {os.path.basename(rp)}")

    # 同时保存含数字ID的对照版 Excel 供自查
    df_merged = pd.merge(
        df_pat[['stock_code', 'transaction_date', 'pattern_type', 'pattern_explanation', 'pattern_id']],
        df_res[['stock_code', 'transaction_date', 'capital_type', 'capital_intention', 'intent_id']],
        on=['stock_code', 'transaction_date'], how='outer'
    )
    df_merged = df_merged[['stock_code', 'transaction_date', 'pattern_id', 'pattern_type',
        'pattern_explanation', 'intent_id', 'capital_type', 'capital_intention']]

    with pd.ExcelWriter(mp, engine='openpyxl') as writer:
        for date in sorted(df_merged['transaction_date'].unique()):
            sheet_df = df_merged[df_merged['transaction_date'] == date].copy()
            sheet_df = sheet_df.sort_values('stock_code').reset_index(drop=True)
            sheet_df.to_excel(writer, sheet_name=str(date), index=False)
    print(f"已保存（含ID对照）: {mp} (共 {df_merged['transaction_date'].nunique()} 个 sheet)")

    if 'cluster_id' not in df_feat.columns:
        print("跳过评估（无 cluster_id）")
        return
    print("===== 离线评估 =====")
    fc = [c for c in df_feat.columns if c not in ('stock_code', 'transaction_date', 'cluster_id')]
    Xs = StandardScaler().fit_transform(df_feat[fc].values)
    nu = df_feat['cluster_id'].nunique()
    if nu > 1 and nu < Xs.shape[0]:
        print(f"Task1: 轮廓系数={silhouette_score(Xs, df_feat['cluster_id']):.4f} "
            f"CH={calinski_harabasz_score(Xs, df_feat['cluster_id']):.4f} "
            f"DB={davies_bouldin_score(Xs, df_feat['cluster_id']):.4f}")
    else:
        print(f"Task1: 聚类数={nu}，样本数={Xs.shape[0]}，跳过评估")
    n = len(df_res)
    print(f"Task2: 游资={(df_res['capital_type'] == '游资').sum() / n * 100:.1f}% "
        f"量化={(df_res['capital_type'] == '量化').sum() / n * 100:.1f}% "
        f"散户={(df_res['capital_type'] == '散户').sum() / n * 100:.1f}%")


def run_baseline(input_path, output_dir):
    print(f"\n{'='*65}")
    print(f"[阶段2/2] 模型分析（Baseline v14 天池标准格式版）")
    print(f"{'='*65}")
    print(f"输入: {input_path}\n输出: {output_dir}\n")

    df_raw = load_and_preprocess_baseline(input_path)
    df_feat = extract_all_feature_baseline(df_raw)
    df_pat = task1_trade_pattern_clustering(df_feat)
    df_res = task2_capital_recognition(df_feat)
    save_results_baseline(df_pat, df_res, df_feat, output_dir)
    print(f"\n✅ 流程完成！打包 {output_dir}/pattern_reco.csv + predict_result.csv 为 submit.zip 提交")
    return df_pat, df_res


# ============================================================
#  PART 3: 主流程整合
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='AFAC2026 赛题一 完整流水线 v14 天池标准格式版')
    parser.add_argument('--input', help='已有训练数据路径（如 all_100.xlsx）')
    parser.add_argument('--batch-dir', help='原始数据目录，先自动生成训练数据')
    parser.add_argument('--output', required=True, help='输出目录')
    parser.add_argument('--merge', action='store_true', help='合并所有数据')
    args = parser.parse_args()

    if args.batch_dir:
        merge_path = generate_train_data(args.batch_dir, args.output, merge=args.merge)
        input_file = merge_path
    elif args.input:
        input_file = args.input
    else:
        print("❌ 请指定 --input 或 --batch-dir")
        print("  --input: 已有训练数据路径")
        print("  --batch-dir: 原始数据目录，先自动生成训练数据")
        sys.exit(1)

    if not os.path.exists(input_file):
        print(f"❌ 输入文件不存在: {input_file}")
        sys.exit(1)

    run_baseline(input_file, args.output)

    print(f"\n{'='*65}")
    print(f"🎉 全部完成！")
    print(f"  训练数据: {input_file}")
    print(f"  结果输出: {os.path.abspath(args.output)}")
    print(f"{'='*65}")


if __name__ == '__main__':
    main()
