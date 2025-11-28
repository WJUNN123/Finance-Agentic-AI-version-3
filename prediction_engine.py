"""
Hybrid prediction engine: LSTM + XGBoost
- Preserves original LSTM-based predictor implementation.
- Adds XGBoost-based predictor.
- Adds HybridPredictor that fuses LSTM + XGBoost forecasts (weighted average).
- Exported API:
    - PricePredictor (LSTM) : unchanged public methods (train_or_load, predict_future, predict)
    - XGBoostPredictor : same interface (train_or_load, predict_future, predict)
    - HybridPredictor : train_or_load, predict_future, predict
    - get_hybrid_predictor(symbol, lstm_weight=0.5, look_back=None, features=None)
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import logging
import joblib
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from io import StringIO
from contextlib import contextmanager

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import xgboost as xgb

from config import Config

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@contextmanager
def suppress_stdout():
    """Context manager to suppress stdout"""
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        yield
    finally:
        sys.stdout = old_stdout


class PricePredictor:
    """Original LSTM model for price prediction with model persistence"""

    def __init__(self, symbol: str, look_back: int = None, features: List[str] = None):
        self.symbol = symbol
        self.look_back = look_back or Config.LSTM_LOOKBACK_DAYS
        self.features = features if features is not None else ['price', 'volume']
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))

        # Setup paths for saving the model and scaler
        self.model_dir = Path("trained_models")
        self.model_dir.mkdir(exist_ok=True)
        self.model_path = self.model_dir / f"{self.symbol}_lstm_model.keras"
        self.scaler_path = self.model_dir / f"{self.symbol}_scaler.joblib"

    def _prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare multivariate data for the LSTM model"""
        missing_features = [f for f in self.features if f not in df.columns]
        if missing_features:
            logger.warning(f"Missing features {missing_features} in DataFrame")
            available_features = [f for f in self.features if f in df.columns]
            if not available_features:
                raise ValueError("No valid features found in DataFrame")
            self.features = available_features

        data = df.filter(self.features)
        data = data.ffill().bfill()
        data = data.replace([np.inf, -np.inf], np.nan).ffill()

        dataset = data.values
        scaled_data = self.scaler.fit_transform(dataset)
        return dataset, scaled_data

    def _create_dataset(self, dataset: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create training dataset from the scaled multivariate data"""
        X, y = [], []
        for i in range(self.look_back, len(dataset)):
            X.append(dataset[i - self.look_back:i, :])
            y.append(dataset[i, 0])  # Predict the first feature (price)
        return np.array(X), np.array(y)

    def _build_model(self, input_shape: Tuple[int, int]):
        """Build the LSTM model using the functional API"""
        inputs = Input(shape=input_shape)
        x = LSTM(128, return_sequences=True)(inputs)
        x = Dropout(0.2)(x)
        x = LSTM(64, return_sequences=False)(x)
        x = Dropout(0.2)(x)
        x = Dense(25)(x)
        outputs = Dense(1)(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='mean_squared_error')
        self.model = model
        logger.info("LSTM model built successfully")

    def train_or_load(self, df: pd.DataFrame, force_retrain: bool = False):
        """Either load a pre-trained model or train a new one"""
        if (self.model_path.exists() and self.scaler_path.exists() and not force_retrain):
            try:
                logger.info(f"Loading pre-trained LSTM model for {self.symbol.upper()}")
                self.model = tf.keras.models.load_model(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                logger.info("LSTM model loaded successfully")
                return
            except Exception as e:
                logger.warning(f"Could not load LSTM model, will retrain. Error: {e}")

        logger.info(f"Training new LSTM model for {self.symbol.upper()}")

        min_required = self.look_back + 20
        if len(df) < min_required:
            logger.warning(
                f"Not enough historical data to train optimally. Required {min_required}, got {len(df)}"
            )
            if len(df) < self.look_back + 5:
                raise ValueError(
                    f"Insufficient data for training. Need at least {self.look_back + 5} rows, got {len(df)}"
                )

        _, scaled_data = self._prepare_data(df)
        X_train, y_train = self._create_dataset(scaled_data)

        if X_train.shape[0] == 0:
            raise RuntimeError("Could not create a training set from the provided data")

        logger.info(f"Preparing LSTM training data: {X_train.shape[0]} samples")
        self._build_model((X_train.shape[1], X_train.shape[2]))

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model_checkpoint = ModelCheckpoint(self.model_path, save_best_only=True, monitor='val_loss')

        logger.info("Training LSTM model...")
        with suppress_stdout():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model.fit(
                    X_train, y_train,
                    batch_size=min(32, max(1, len(X_train)//4)),
                    epochs=50,
                    validation_split=0.1,
                    callbacks=[early_stopping, model_checkpoint],
                    verbose=0
                )

        joblib.dump(self.scaler, self.scaler_path)
        if self.model is None or self.scaler is None:
            raise RuntimeError(f"LSTM model for {self.symbol} was not trained or loaded successfully")

    def predict_future(self, df: pd.DataFrame, days: int = None) -> Optional[List[Dict]]:
        """Predict future prices using the trained multivariate LSTM model"""
        days = days or Config.FORECAST_DAYS
        if self.model is None:
            logger.error("LSTM model is not loaded. Please call train_or_load() first")
            return None

        try:
            data = df.filter(self.features)
            data = data.ffill().bfill()
            data = data.replace([np.inf, -np.inf], np.nan).ffill()
            if len(data) < self.look_back:
                logger.error(f"Not enough data for LSTM prediction. Need {self.look_back}, got {len(data)}")
                return None

            scaled_data = self.scaler.transform(data.values)
            last_sequence = scaled_data[-self.look_back:]
            predictions = []

            logger.info(f"LSTM generating {days}-day forecast...")
            for day in range(days):
                current_sequence_reshaped = np.reshape(last_sequence, (1, self.look_back, len(self.features)))
                with suppress_stdout():
                    predicted_scaled_price = self.model.predict(current_sequence_reshaped, verbose=0)[0][0]
                predictions.append(predicted_scaled_price)

                if len(self.features) > 1:
                    next_step_features = last_sequence[-1, 1:].copy()
                    new_row = np.insert(next_step_features, 0, predicted_scaled_price)
                else:
                    new_row = np.array([predicted_scaled_price])

                last_sequence = np.vstack([last_sequence[1:], new_row.reshape(1,-1)])

            dummy_array = np.zeros((len(predictions), len(self.features)))
            dummy_array[:,0] = predictions
            predicted_prices_unscaled = self.scaler.inverse_transform(dummy_array)[:,0]

            forecast = []
            for i, price in enumerate(predicted_prices_unscaled):
                forecast.append({
                    'date': (datetime.now() + timedelta(days=i+1)).strftime('%Y-%m-%d'),
                    'predicted_price': round(max(price, 0.01),4)
                })
            return forecast
        except Exception as e:
            logger.error(f"LSTM prediction failed: {e}")
            return None

    # Backwards compatible alias
    def predict(self, df: pd.DataFrame, days:int=None):
        return self.predict_future(df, days)


class XGBoostPredictor:
    """XGBoost model for short-term price prediction"""

    def __init__(self, symbol: str, look_back: int = None, features: List[str] = None):
        self.symbol = symbol
        self.look_back = look_back or Config.LSTM_LOOKBACK_DAYS
        self.features = features if features is not None else ['price', 'volume']
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0,1))

        self.model_dir = Path("trained_models")
        self.model_dir.mkdir(exist_ok=True)
        self.model_path = self.model_dir / f"{self.symbol}_xgb_model.joblib"
        self.scaler_path = self.model_dir / f"{self.symbol}_xgb_scaler.joblib"

    def _prepare_supervised(self, df: pd.DataFrame) -> Tuple[np.ndarray,np.ndarray]:
        data = df.filter(self.features).ffill().bfill()
        scaled = self.scaler.fit_transform(data.values)
        X, y = [], []
        for i in range(self.look_back, len(scaled)):
            X.append(scaled[i-self.look_back:i].flatten())
            y.append(scaled[i,0])
        return np.array(X), np.array(y)

    def train_or_load(self, df: pd.DataFrame, force_retrain: bool=False):
        if self.model_path.exists() and self.scaler_path.exists() and not force_retrain:
            try:
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                return
            except Exception:
                logger.info("Could not load existing XGBoost model/scaler; will retrain.")

        X, y = self._prepare_supervised(df)
        if len(X) == 0:
            raise RuntimeError("Not enough data to train XGBoost model")
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, shuffle=False)
        model = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05,
                                 objective='reg:squarederror', verbosity=0, n_jobs=1)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=False)
        self.model = model
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)

    def predict_future(self, df: pd.DataFrame, days:int=None) -> Optional[List[Dict]]:
        """
        Returns a list of dicts to match LSTM output:
        [{ 'date': 'YYYY-MM-DD', 'predicted_price': float }, ...]
        """
        days = days or Config.FORECAST_DAYS
        if self.model is None:
            logger.error("XGBoost model is not loaded. Please call train_or_load() first")
            return None

        data = df.filter(self.features).ffill().bfill()
        if len(data) < self.look_back:
            logger.error(f"Not enough data for XGBoost prediction. Need {self.look_back}, got {len(data)}")
            return None

        scaled = self.scaler.transform(data.values)
        last_window = scaled[-self.look_back:]
        predictions_scaled = []
        for _ in range(days):
            X_input = last_window.flatten().reshape(1,-1)
            pred_scaled = float(self.model.predict(X_input)[0])
            predictions_scaled.append(pred_scaled)
            if len(self.features)>1:
                next_row = np.insert(last_window[-1,1:],0,pred_scaled)
            else:
                next_row = np.array([pred_scaled])
            last_window = np.vstack([last_window[1:], next_row.reshape(1,-1)])

        dummy = np.zeros((len(predictions_scaled), len(self.features)))
        dummy[:,0] = predictions_scaled
        predicted_prices = self.scaler.inverse_transform(dummy)[:,0].tolist()

        forecast = []
        for i, price in enumerate(predicted_prices):
            forecast.append({
                'date': (datetime.now()+timedelta(days=i+1)).strftime('%Y-%m-%d'),
                'predicted_price': round(max(price,0.01),4)
            })
        return forecast

    # Backwards compatible alias
    def predict(self, df: pd.DataFrame, days:int=None):
        return self.predict_future(df, days)


class HybridPredictor:
    """Hybrid predictor combining LSTM + XGBoost"""

    def __init__(self, symbol: str, lstm_weight: float = 0.5, look_back: int = None, features: List[str] = None):
        if not 0<=lstm_weight<=1:
            raise ValueError("lstm_weight must be between 0 and 1")
        self.symbol = symbol
        self.lstm_weight = lstm_weight
        self.xgb_weight = 1.0 - lstm_weight
        self.lstm = PricePredictor(symbol, look_back, features)
        self.xgb = XGBoostPredictor(symbol, look_back, features)

    def train_or_load(self, df: pd.DataFrame, force_retrain: bool=False):
        logger.info(f"Hybrid: train_or_load for {self.symbol.upper()}")
        # Train/load both models (if not available)
        self.lstm.train_or_load(df, force_retrain)
        self.xgb.train_or_load(df, force_retrain)

    def predict_future(self, df: pd.DataFrame, days: int=None) -> Optional[List[Dict]]:
        days = days or Config.FORECAST_DAYS

        lstm_forecast = self.lstm.predict_future(df, days)
        xgb_forecast = self.xgb.predict_future(df, days)

        if lstm_forecast is None or xgb_forecast is None:
            logger.error("One of the component predictors returned None")
            return None

        # Extract numeric prices from both forecasts (they are lists of dicts)
        lstm_prices = [entry['predicted_price'] if isinstance(entry, dict) and 'predicted_price' in entry else float(entry) for entry in lstm_forecast]
        xgb_prices = [entry['predicted_price'] if isinstance(entry, dict) and 'predicted_price' in entry else float(entry) for entry in xgb_forecast]

        # ensure lengths match (zip will truncate to shortest)
        fused = [(self.lstm_weight * l + self.xgb_weight * x) for l, x in zip(lstm_prices, xgb_prices)]

        forecast = []
        for i, price in enumerate(fused):
            forecast.append({
                'date': (datetime.now()+timedelta(days=i+1)).strftime('%Y-%m-%d'),
                'predicted_price': round(max(price,0.01),4)
            })
        return forecast

    # Backwards compatible alias
    def predict(self, df: pd.DataFrame, days:int=None):
        return self.predict_future(df, days)


def get_hybrid_predictor(symbol: str, lstm_weight: float = 0.5, look_back: int=None, features:List[str]=None) -> HybridPredictor:
    """Factory to obtain a hybrid predictor"""
    return HybridPredictor(symbol, lstm_weight, look_back, features)


class PredictionAdjuster:
    """Adjusts raw forecast based on sentiment and technical signals (kept unchanged)"""

    def adjust(self, raw_forecast: List[Dict], decision_data: Optional[Dict]=None,
               confidence_data: Optional[Dict]=None, current_price: Optional[float]=None) -> List[Dict]:
        """
        Backwards-compatible adjust():
        - Old callers may pass (raw_forecast, market_data)
        - New callers may pass (raw_forecast, decision_data, confidence_data, current_price)
        This function will detect and adapt.
        """
        # Backwards compatibility: if only raw_forecast and market_data passed
        if confidence_data is None and current_price is None and isinstance(decision_data, dict):
            market_data = decision_data
            # best-effort extraction of current price
            try:
                current_price = market_data.get('market_data', {}).get('current_price') or market_data.get('current_price')
            except Exception:
                current_price = None
            # Provide safe defaults for missing pieces
            decision_data = {}
            confidence_data = {"overall_confidence": 50}

        # If still missing current_price, try to set to the first forecast price
        if current_price is None:
            try:
                current_price = raw_forecast[0]['predicted_price'] if raw_forecast and isinstance(raw_forecast[0], dict) else 0.0
            except Exception:
                current_price = 0.0

        if not raw_forecast:
            return []

        signal_strength = (decision_data.get('signal_strength', 0) if isinstance(decision_data, dict) else 0) / 100.0
        confidence = (confidence_data.get('overall_confidence', 50) if isinstance(confidence_data, dict) else 50) / 100.0
        adjustment_factor = signal_strength * confidence * 0.5

        adjusted_forecast = []
        last_price = current_price
        for forecast_point in raw_forecast:
            raw_predicted_price = forecast_point.get('predicted_price') if isinstance(forecast_point, dict) else float(forecast_point)
            model_change = raw_predicted_price - last_price
            adjusted_change = model_change * (1 + adjustment_factor)
            adjusted_price = last_price + adjusted_change
            adjusted_forecast.append({
                'date': forecast_point.get('date') if isinstance(forecast_point, dict) and 'date' in forecast_point
                        else (datetime.now()+timedelta(days=len(adjusted_forecast)+1)).strftime('%Y-%m-%d'),
                'predicted_price': round(max(adjusted_price, 0.01), 2)
            })
            last_price = adjusted_price

        return adjusted_forecast
