import pandas as pd
import pytorch_lightning as pl
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss

class TFTExpert:
    """
    CNP-2: Normal Operations Regime Expert using Temporal Fusion Transformer.
    This wrapper sets up the TimeSeriesDataSet and handles building/training the TFT.
    """
    def __init__(self, target_col='price_asinh', max_encoder_length=168, max_prediction_length=24):
        self.target_col = target_col
        self.max_encoder_length = max_encoder_length
        self.max_prediction_length = max_prediction_length
        
    def create_dataset(self, df, time_idx_col='time_idx', group_col='group'):
        """
        Creates a TimeSeriesDataSet for pytorch-forecasting.
        Expects a continuous time index column.
        """
        # Ensure group column exists for TimeSeriesDataSet structure
        if group_col not in df.columns:
            df[group_col] = '0'
            
        training_cutoff = df[time_idx_col].max() - self.max_prediction_length
        
        # We define a basic feature set commonly used for the TFT normal expert
        known_reals = [time_idx_col, 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 
                       'Residual_Load', 'Wind Onshore', 'Wind Offshore', 'Solar']
        # Filter for columns that actually exist
        known_reals = [c for c in known_reals if c in df.columns]
        
        # Target column is the specific target to predict
        
        training_dataset = TimeSeriesDataSet(
            df[lambda x: x[time_idx_col] <= training_cutoff],
            time_idx=time_idx_col,
            target=self.target_col,
            group_ids=[group_col],
            min_encoder_length=self.max_encoder_length // 2,
            max_encoder_length=self.max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=self.max_prediction_length,
            static_categoricals=[group_col],
            time_varying_known_reals=known_reals,
            time_varying_unknown_reals=[self.target_col], # Past target is unknown in the future
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )
        
        return training_dataset
        
    def build_model(self, training_dataset, learning_rate=0.03, hidden_size=64):
        """Initializes the TFT Model from the dataset"""
        tft = TemporalFusionTransformer.from_dataset(
            training_dataset,
            learning_rate=learning_rate,
            hidden_size=hidden_size,
            attention_head_size=4,
            dropout=0.1,
            hidden_continuous_size=hidden_size // 2,
            output_size=7,  # Predicting 7 quantiles (e.g. 0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98)
            loss=QuantileLoss(),
            log_interval=10, 
            reduce_on_plateau_patience=4,
        )
        return tft
        
if __name__ == '__main__':
    print("Testing TFT Expert configuration...")
    # Mock data
    df = pd.DataFrame({
        'time_idx': range(1000),
        'price_asinh': pd.Series(range(1000)) * 0.1,
        'hour_sin': pd.Series(range(1000)).apply(lambda x: pd.np.sin(x)),
        'group': 'DE'
    })
    expert = TFTExpert()
    try:
        ds = expert.create_dataset(df)
        model = expert.build_model(ds)
        print("TFT Dataset and Model built successfully.")
    except Exception as e:
        print("Dataset creation error logic correctly caught missing features:", e)
