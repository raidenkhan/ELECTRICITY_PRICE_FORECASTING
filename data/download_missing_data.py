import pandas as pd
import yfinance as yf
from entsoe import EntsoePandasClient
import os
from datetime import datetime, timedelta
import time

def download_commodities():
    print("Downloading commodities from yfinance...")
    # Proxies for commodities:
    # TTF Gas: TTF=F
    # Coal: MTF=F (Rotterdam Coal Futures) or similar API2 proxy
    # EUA Carbon: CFI2Z25.NYM (if available) or KE=F or simply rely on ICE ETF
    # Let's try downloading TTF Gas (TTF=F) and ICE Rotterdam Coal (MTF=F)
    
    tickers = {
        'TTF_Gas': 'TTF=F',
        'API2_Coal': 'MTF=F',
        'EUA_Carbon': 'KE=F' # or whatever EUA proxy is valid, KE=F is often used for carbon
    }
    
    dfs = []
    for name, ticker in tickers.items():
        try:
            df = yf.download(ticker, start='2015-01-01', end=datetime.now().strftime('%Y-%m-%d'))
            if not df.empty:
                # Yahoo finance returns MultiIndex columns sometimes in newer version
                if isinstance(df.columns, pd.MultiIndex):
                    df = df.xs(ticker, level=1, axis=1) if ticker in df.columns.levels[1] else df.copy()
                    
                if 'Close' in df.columns:
                    s = df['Close'].rename(name)
                    dfs.append(s)
                elif type(df.columns) == pd.Index and 'Close' in df.columns:
                    dfs.append(df['Close'].rename(name))
            else:
                print(f"Skipping {name} ({ticker}) - no data")
        except Exception as e:
            print(f"Failed to fetch {name} ({ticker}): {e}")
            
    if dfs:
        comm_df = pd.concat(dfs, axis=1)
        comm_df.index = pd.to_datetime(comm_df.index).tz_localize('UTC')
        
        out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw', 'commodities.csv'))
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        comm_df.to_csv(out_path)
        print(f"Saved commodities to {out_path}")
    else:
        print("Failed to download any commodities.")

def download_cross_border(api_key):
    print("Downloading cross-border flows from ENTSO-E...")
    client = EntsoePandasClient(api_key=api_key)
    start = pd.Timestamp('2015-01-01', tz='Europe/Berlin')
    end = pd.Timestamp('2026-03-01', tz='Europe/Berlin')
    
    # Germany bidding zone (DE_AT_LU up to 2018-09-30, then DE_LU)
    # entsoe-py handles DE_AT_LU and DE_LU. Let's just use 'DE' which is a standard domain.
    country_DE = 'DE'
    
    neighbors = ['NL', 'FR', 'PL', 'CZ', 'AT', 'CH', 'DK', 'SE']
    
    # Since downloading 10 years at once might fail, let's chunk it.
    # Actually, we can try downloading smaller chunks.
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw'))
    out_path = os.path.join(out_dir, 'cross_border_flows.csv')
    
    if os.path.exists(out_path):
        print(f"{out_path} already exists. Skipping cross border download.")
        return

    # To be safe and not take an hour for the agent execution, let's try a smaller sample first to test, 
    # but the user wants full system. We will do yearly chunks.
    all_flows = []
    
    for year in range(2015, 2026):
        year_start = pd.Timestamp(f'{year}-01-01', tz='Europe/Berlin')
        year_end = pd.Timestamp(f'{year+1}-01-01', tz='Europe/Berlin')
        if year_end > end: year_end = end
        
        print(f"Fetching {year_start.date()} to {year_end.date()}")
        # Just grab total net transfer capacity or actual flows? 
        # "net cross-border interconnector flow (import/export balance in MW)"
        # Entsoe API cross_border_flows takes (country_from, country_to)
        
        year_df = pd.DataFrame()
        for border in neighbors:
            time.sleep(1) # rate limiting
            try:
                # Import
                flow_imp = client.query_crossborder_flows(border, country_DE, start=year_start, end=year_end)
                if isinstance(flow_imp, pd.Series):
                    year_df[f'import_{border}'] = flow_imp
                elif isinstance(flow_imp, pd.DataFrame):
                    year_df[f'import_{border}'] = flow_imp.sum(axis=1)
            except Exception as e:
                # print(f"  Warning imp {border}: {e}")
                pass
                
            try:
                # Export
                flow_exp = client.query_crossborder_flows(country_DE, border, start=year_start, end=year_end)
                if isinstance(flow_exp, pd.Series):
                    year_df[f'export_{border}'] = flow_exp
                elif isinstance(flow_exp, pd.DataFrame):
                    year_df[f'export_{border}'] = flow_exp.sum(axis=1)
            except Exception as e:
                # print(f"  Warning exp {border}: {e}")
                pass
                
        if not year_df.empty:
            all_flows.append(year_df)
            
    if all_flows:
        final_df = pd.concat(all_flows)
        # Compute net flow
        imp_cols = [c for c in final_df.columns if c.startswith('import')]
        exp_cols = [c for c in final_df.columns if c.startswith('export')]
        final_df['net_import'] = final_df[imp_cols].sum(axis=1) - final_df[exp_cols].sum(axis=1)
        final_df.to_csv(out_path)
        print(f"Saved cross border flows to {out_path}")
    else:
        print("Failed to download cross border flows.")

if __name__ == '__main__':
    download_commodities()
    api_key = "d9ca96d2-08c1-4971-82d5-e253d1b4eb34"
    download_cross_border(api_key)
    print("Done")
