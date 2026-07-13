import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
from src.data.preprocess import EPFPreprocessor
from src.models.ds_hdp_hmm import VariationalStickyHMM
from src.models.experts.evt_expert import EVTExpert, SolarConditionedExpert
from src.models.experts.tft_expert import TFTExpert
from src.models.mixture_model import EPFMixtureCombiner

class EPFPipeline:
    def __init__(self, data_path, comm_path=None, flow_path=None):
        self.data_path = data_path
        self.comm_path = comm_path
        self.flow_path = flow_path
        self.preprocessor = EPFPreprocessor()
        
    def load_and_merge(self):
        print("Loading raw datasets...")
        df = pd.read_csv(self.data_path)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], utc=True)
            df = df.set_index('date').sort_index()
            
        if self.comm_path and os.path.exists(self.comm_path):
            comm = pd.read_csv(self.comm_path, index_col=0, parse_dates=True)
            comm.index = pd.to_datetime(comm.index, utc=True)
            comm = comm.resample('h').ffill()
            df = df.join(comm, how='left').ffill()
            
        if self.flow_path and os.path.exists(self.flow_path):
            flows = pd.read_csv(self.flow_path, index_col=0, parse_dates=True)
            flows.index = pd.to_datetime(flows.index, utc=True)
            df = df.join(flows, how='left')
            
        # Re-ensure index is DatetimeIndex (joins can demote dtype)
        df.index = pd.DatetimeIndex(pd.to_datetime(df.index, utc=True))
        df = df.fillna(0)
        print(f"Data joined. Shape: {df.shape}")
        return df

    def train_hmm(self, process_df, epochs=50):
        print("Extracting daily features for DS-HDP-HMM...")
        daily_stats = self.preprocessor.extract_daily_hmm_features(process_df)
        
        # Features: [price_median_asinh, price_iqr_asinh, dark_doldrums_daily_max, daily_neg_streak, daily_max_range]
        obs = torch.tensor(daily_stats.values, dtype=torch.float32)
        
        print("Training DS-HDP-HMM Custom PyTorch Proxy...")
        K = 10 # Allow up to 10 regimes
        self.hmm = VariationalStickyHMM(obs_dim=obs.shape[1], K_max=K)
        optimizer = torch.optim.Adam(self.hmm.parameters(), lr=0.02)
        
        for ep in range(epochs):
            optimizer.zero_grad()
            loss = self.hmm.compute_loss(obs)
            loss.backward()
            optimizer.step()
            
        print("HMM trained. Extracting Viterbi states...")
        states = self.hmm.viterbi(obs)
        daily_stats['regime'] = states.numpy()
        
        # Map daily regimes back to hourly dataframe
        process_df['date_only'] = process_df.index.date
        daily_stats['date_only'] = daily_stats.index.date
        regime_map = daily_stats.set_index('date_only')['regime'].to_dict()
        process_df['regime'] = process_df['date_only'].map(regime_map)
        
        return process_df

    def construct_experts(self, process_df):
        print("Analyzing discovered regimes to map experts...")
        # Simple heuristic mapping for dynamic regimes:
        # High Variance/Spikes -> EVT (CNP-1)
        # Negative prices > 5% -> Solar (CNP-3)
        # Else -> TFT (CNP-2)
        
        self.expert_map = {}
        for k in process_df['regime'].dropna().unique():
            sub = process_df[process_df['regime'] == k]
            neg_frac = (sub['price'] < 0).mean()
            price_99 = sub['price'].quantile(0.99)
            
            if price_99 > 200:
                self.expert_map[k] = 'evt'
            elif neg_frac > 0.05:
                self.expert_map[k] = 'solar'
            else:
                self.expert_map[k] = 'tft'
                
        print("Expert mapping:", self.expert_map)
        
        # We would then build datasets for TFT and PyTorch loaders for EVT/Solar
        # Here we initialize the models
        
        # Determine feature size for MLP experts
        # We'll use 15 basic features
        feature_cols = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'Residual_Load', 
                       'total_re_penetration', 'dark_doldrums', '12h_price_range', 
                       '3h_neg_streak', 'price_lag_24', 'price_lag_48', 'price_lag_168']
        
        # filter available
        self.ml_features = [c for c in feature_cols if c in process_df.columns]
        
        self.experts = {
            'evt': EVTExpert(len(self.ml_features)),
            'solar': SolarConditionedExpert(len(self.ml_features)),
            'tft': TFTExpert() # TFT is complicated (PyTorch Lightning trainer required)
        }
        self.combiner = EPFMixtureCombiner(self.expert_map)
        print("Pipeline architecture instantiated.")
        
    def run_pipeline(self):
        df = self.load_and_merge()
        df = self.preprocessor.process(df)
        df_regimes = self.train_hmm(df)
        self.construct_experts(df_regimes)
        print("Done. Ready for training and inference phases.")

if __name__ == '__main__':
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw'))
    pipeline = EPFPipeline(
        data_path=os.path.join(data_dir, 'Germany_master_entsoe_2015_2026.csv'),
        comm_path=os.path.join(data_dir, 'commodities.csv'),
        flow_path=os.path.join(data_dir, 'cross_border_flows.csv')
    )
    # Fast test run without actually training TFT for 10 hours
    # Just passing through structure
    pipeline.run_pipeline()
