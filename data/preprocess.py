import pandas as pd
import numpy as np
from datetime import datetime
import math

class EPFPreprocessor:
    def __init__(self, price_col='price', scaling_constant=50.0):
        self.price_col = price_col
        self.scaling_constant = scaling_constant

    def transform_price(self, df):
        """Asinh transform: p_tilde = sinh^-1(p / c)"""
        df = df.copy()
        df['price_asinh'] = np.arcsinh(df[self.price_col] / self.scaling_constant)
        return df

    def inverse_transform_price(self, p_tilde):
        """Inverse asinh transform: p = c * sinh(p_tilde)"""
        return self.scaling_constant * np.sinh(p_tilde)

    def compute_residual_load(self, df, load_col='Actual Load', 
                              wind_on='Wind Onshore', wind_off='Wind Offshore', 
                              solar='Solar'):
        df = df.copy()
        # Fallback column names handling
        if wind_on not in df.columns and 'Wind Onshore.1' in df.columns:
            wind_on = 'Wind Onshore.1'
        
        wind_gen = df[wind_on].fillna(0) + df[wind_off].fillna(0)
        solar_gen = df[solar].fillna(0)
        df['Residual_Load'] = df[load_col] - wind_gen - solar_gen
        df['Total_Wind_Solar'] = wind_gen + solar_gen
        return df

    def capacity_normalization(self, df):
        """
        Wind and solar capacity normalization.
        Since explicit installed capacity might not be in the raw CSV,
        we estimate it using a rolling max window (1 year = 8760 hours)
        to capture the dynamic installed capacity.
        """
        df = df.copy()
        for col, new_col in [('Wind Onshore', 'wind_on_penetration'), 
                             ('Wind Offshore', 'wind_off_penetration'),
                             ('Solar', 'solar_penetration')]:
            target_col = col if col in df.columns else (col + '.1' if col + '.1' in df.columns else None)
            if target_col:
                # Rolling 1-year max to estimate installed capacity
                capacity = df[target_col].rolling(window=8760, min_periods=1).max()
                # Forward fill from max
                capacity = capacity.cummax()
                # Prevent div zero
                capacity = capacity.replace(0, 1)
                df[new_col] = df[target_col] / capacity
                
        # Total wind + solar penetration
        tot_gen = df.get('Wind Onshore', 0) + df.get('Wind Offshore', 0) + df.get('Solar', 0)
        # 1-year max of total
        tot_cap = tot_gen.rolling(window=8760, min_periods=1).max().cummax().replace(0, 1)
        if isinstance(tot_cap, pd.Series):
             df['total_re_penetration'] = (tot_gen / tot_cap).fillna(0)
        return df

    def cyclic_time_features(self, df):
        """Sine/Cosine encoding of time attributes"""
        df = df.copy()
        if 'date' in df.columns:
            dt = pd.DatetimeIndex(pd.to_datetime(df['date']))
        else:
            dt = pd.DatetimeIndex(pd.to_datetime(df.index))
            
        hour = dt.hour
        dayofweek = dt.dayofweek
        dayofyear = dt.dayofyear
        
        df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        df['dow_sin'] = np.sin(2 * np.pi * dayofweek / 7)
        df['dow_cos'] = np.cos(2 * np.pi * dayofweek / 7)
        df['doy_sin'] = np.sin(2 * np.pi * dayofyear / 365.25)
        df['doy_cos'] = np.cos(2 * np.pi * dayofyear / 365.25)
        
        # Simple solar position proxy for Germany (lat ~51)
        # Sunrise/sunset minutes from midnight approximation
        # Base approximation mapping doy to sunrise/sunset
        declination = 23.45 * np.sin(2 * np.pi * (284 + dayofyear) / 365)
        # hour angle
        tan_lat = np.tan(np.radians(51.0))
        tan_dec = np.tan(np.radians(declination))
        # Ensure domain is valid
        val = -tan_lat * tan_dec
        val = np.clip(val, -1, 1)
        hour_angle = np.degrees(np.arccos(val))
        
        # Solar noon approx 12:00 UTC
        sunrise_hour = 12.0 - (hour_angle / 15.0)
        sunset_hour = 12.0 + (hour_angle / 15.0)
        
        df['sunrise_minutes'] = sunrise_hour * 60
        df['sunset_minutes'] = sunset_hour * 60
        
        return df

    def regime_conditioning_features(self, df):
        """
        Features used in the DS-HDP-HMM emission layer:
        - 7-day rolling average of wind+solar penetration (dark doldrums)
        - 12-hour price range (spikes)
        - 3-consecutive-hour negative price indicator
        """
        df = df.copy()
        
        # 1. 7-day rolling wind+solar penetration
        if 'total_re_penetration' in df.columns:
            df['7d_re_penetration'] = df['total_re_penetration'].rolling(window=24*7, min_periods=1).mean()
            df['dark_doldrums'] = (df['7d_re_penetration'] < 0.20).astype(int)
        
        # 2. 12-hour price range
        df['12h_price_range'] = df[self.price_col].rolling(window=12, min_periods=1).max() - \
                                df[self.price_col].rolling(window=12, min_periods=1).min()
                                
        # 3. 3-consecutive-hour negative price indicator
        is_neg = (df[self.price_col] < 0).astype(int)
        streak = is_neg.rolling(window=3).sum()
        df['3h_neg_streak'] = (streak == 3).astype(int)
        
        return df
        
    def gas_crisis_indicator(self, df):
        """Structural break proxy (2021-2022)"""
        df = df.copy()
        if 'date' in df.columns:
            dt = pd.DatetimeIndex(pd.to_datetime(df['date']))
        else:
            dt = pd.DatetimeIndex(pd.to_datetime(df.index))
            
        is_crisis = ((dt >= '2021-06-01') & (dt <= '2023-03-01')).astype(int)
        df['gas_crisis_regime'] = is_crisis
        # Interaction with gas price handled downstream
        return df

    def process(self, df):
        """Run full preprocessing pipeline on hourly data"""
        df = self.transform_price(df)
        df = self.compute_residual_load(df)
        df = self.capacity_normalization(df)
        df = self.cyclic_time_features(df)
        df = self.regime_conditioning_features(df)
        df = self.gas_crisis_indicator(df)
        
        # Add lagged prices (t-24, t-48, t-168)
        for lag in [24, 48, 168]:
            df[f'price_lag_{lag}'] = df[self.price_col].shift(lag)
            
        return df

    def extract_daily_hmm_features(self, df):
        """
        Extract daily summary observations to feed into the DS-HDP-HMM.
        Expects a datetime index.
        """
        if 'date' in df.columns:
            df = df.set_index(pd.to_datetime(df['date']))
            
        daily = df.resample('D').agg({
            self.price_col: ['median', lambda x: x.quantile(0.75) - x.quantile(0.25)],
            'dark_doldrums': 'max',
            '3h_neg_streak': 'max',
            '12h_price_range': 'max'
        })
        daily.columns = ['price_median', 'price_iqr', 'dark_doldrums_daily_max', 
                         'daily_neg_streak', 'daily_max_range']
                         
        # asinh transform the daily median and IQR for stability
        daily['price_median_asinh'] = np.arcsinh(daily['price_median'] / self.scaling_constant)
        daily['price_iqr_asinh'] = np.arcsinh(daily['price_iqr'] / self.scaling_constant)
        
        return daily.dropna()

if __name__ == '__main__':
    print("Testing Preprocessor...")
    df = pd.DataFrame({
        'date': pd.date_range('2025-01-01', periods=48, freq='H'),
        'price': np.random.normal(50, 100, 48),
        'Actual Load': np.random.normal(60000, 5000, 48),
        'Wind Onshore': np.random.normal(15000, 2000, 48),
        'Wind Offshore': np.random.normal(5000, 1000, 48),
        'Solar': np.random.normal(5000, 5000, 48).clip(0)
    })
    preprocessor = EPFPreprocessor()
    df_processed = preprocessor.process(df)
    
    daily = preprocessor.extract_daily_hmm_features(df_processed)
    print("Hourly features shape:", df_processed.shape)
    print("Daily HMM observations shape:", daily.shape)
