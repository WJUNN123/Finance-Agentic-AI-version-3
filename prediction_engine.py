"""
Hybrid prediction engine: LSTM + XGBoost
- Preserves original LSTM-based predictor implementation.
- Adds XGBoost-based predictor.
- Adds HybridPredictor that fuses LSTM + XGBoost forecasts (weighted average).
- Exported API:
    - PricePredictor (LSTM) : unchanged public methods
    - XGBoostPredictor : same interface (train_or_load, predict_future)
    - HybridPredictor : train_or_load, predict_future
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

from tensorflow.keras.models import Sequential, Model
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
    """Original LSTM model for price prediction with model persistence (kept mostly unchanged)"""

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

        # Handle any NaN or infinite values
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
        if (self.model_path.exists() and self.scaler_path.exists()
                and not force_retrain):
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
                f"Not enough historical data to train optimally. "
                f"Required {min_required}, got {len(df)}"
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

        # Callbacks for training
        early_stopping = EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
        model_checkpoint = ModelCheckpoint(
            self.model_path, save_best_only=True, monitor='val_loss'
        )

        logger.info("Training LSTM model...")

        with suppress_stdout():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model.fit(
                    X_train, y_train,
                    batch_size=min(32, max(1, len(X_train) // 4)),
                    epochs=50,
                    validation_split=0.1,
                    callbacks=[early_stopping, model_checkpoint],
                    verbose=0
                )

        logger.info(f"LSTM model training completed for {self.symbol.upper()}")

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

                last_sequence = np.vstack([last_sequence[1:], new_row.reshape(1, -1)])

            dummy_array = np.zeros((len(predictions), len(self.features)))
            dummy_array[:, 0] = predictions
            predicted_prices_unscaled = self.scaler.inverse_transform(dummy_array)[:, 0]

            forecast = []
            for i, price in enumerate(predicted_prices_unscaled):
                forecast.append({
                    'date': (datetime.now() + timedelta(days=i + 1)).strftime('%Y-%m-%d'),
                    'predicted_price': round(max(price, 0.01), 4)
                })

            return forecast

        except Exception as e:
            logger.error(f"LSTM prediction failed: {e}")
            return None


class XGBoostPredictor:
    """XGBoost model for short-term price prediction"""

    def __init__(self, symbol: str, look_back: int = None, features: List[str] = None):
        self.symbol = symbol
        self.look_back = look_back or Config.LSTM_LOOKBACK_DAYS
        self.features = features if features is not None else ['price', 'volume']
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))

        # Persist
        self.model_dir = Path("trained_models")
        self.model_dir.mkdir(exist_ok=True)
        self.model_path = self.model_dir / f"{self.symbol}_xgb_model.joblib"
        self.scaler_path = self.model_dir / f"{self.symbol}_xgb_scaler.joblib"

    def _prepare_supervised(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Create supervised dataset for XGBoost by flattening last look_back windows"""
        missing_features = [f for f in self.features if f not in df.columns]
        if missing_features:
            logger.warning(f"Missing features {missing_features} in DataFrame for XGBoost")
            available_features = [f for f in self.features if f in df.columns]
            if not available_features:
                raise ValueError("No valid features for XGBoost found in DataFrame")
            self.features = available_features

        data = df.filter(self.features)
        data = data.ffill().bfill()
        data = data.replace([np.inf, -np.inf], np.nan).ffill()

        values = data.values
        # scale features
        scaled = self.scaler.fit_transform(values)

        X, y = [], []
        for i in range(self.look_back, len(scaled)):
            window = scaled[i - self.look_back:i, :].flatten()  # flatten look_back*features
            X.append(window)
            y.append(scaled[i, 0])  # next day's price (scaled)
        return np.array(X), np.array(y)

    def train_or_load(self, df: pd.DataFrame, force_retrain: bool = False):
        """Train or load XGBoost model"""
        if (self.model_path.exists() and self.scaler_path.exists() and not force_retrain):
            try:
                logger.info(f"Loading pre-trained XGBoost model for {self.symbol.upper()}")
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                logger.info("XGBoost model loaded successfully")
                return
            except Exception as e:
                logger.warning(f"Could not load XGBoost model, will retrain. Error: {e}")

        logger.info(f"Training new XGBoost model for {self.symbol.upper()}")

        min_required = self.look_back + 20
        if len(df) < min_required:
            logger.warning(f"Not enough data to train XGBoost. Required {min_required}, got {len(df)}")
            if len(df) < self.look_back + 5:
                raise ValueError(f"Insufficient data for XGBoost training. Need at least {self.look_back + 5} rows, got {len(df)}")

        X, y = self._prepare_supervised(df)

        if X.shape[0] == 0:
            raise RuntimeError("Could not create XGBoost training set from provided data")

        # simple train/test split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, shuffle=False)

        # XGBoost regressor
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            objective='reg:squarederror',
            verbosity=0,
            n_jobs=1
        )

        logger.info("Fitting XGBoost model...")
        with suppress_stdout():
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=False)

        self.model = model
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        logger.info(f"XGBoost model training completed for {self.symbol.upper()}")

    def predict_future(self, df: pd.DataFrame, days: int = None) -> Optional[List[Dict]]:
        """Iterative prediction using XGBoost on flattened windows"""
        days = days or Config.FORECAST_DAYS

        if self.model is None:
            logger.error("XGBoost model is not loaded. Please call train_or_load() first")
            return None

        try:
            data = df.filter(self.features)
            data = data.ffill().bfill()
            data = data.replace([np.inf, -np.inf], np.nan).ffill()

            if len(data) < self.look_back:
                logger.error(f"Not enough data for XGBoost prediction. Need {self.look_back}, got {len(data)}")
                return None

            scaled = self.scaler.transform(data.values)
            last_window = scaled[-self.look_back:, :].copy()
            predictions_scaled = []

            logger.info(f"XGBoost generating {days}-day forecast...")

            for day in range(days):
                X_input = last_window.flatten().reshape(1, -1)
                with suppress_stdout():
                    pred_scaled = float(self.model.predict(X_input)[0])
                predictions_scaled.append(pred_scaled)

                # create next window: drop first row, append predicted day's features
                if len(self.features) > 1:
                    next_features = last_window[-1, 1:].copy()
                    new_row = np.insert(next_features, 0, pred_scaled)
                else:
                    new_row = np.array([pred_scaled])

                last_window = np.vstack([last_window[1:], new_row.reshape(1, -1)])

            # convert back to original scale
            dummy = np.zeros((len(predictions_scaled), len(self.features)))
            dummy[:, 0] = predictions_scaled
            predicted_prices_unscaled = self.scaler.inverse_transform(dummy)[:, 0]

            forecast = []
            for i, price in enumerate(predicted_prices_unscaled):
                forecast.append({
                    'date': (datetime.now() + timedelta(days=i + 1)).strftime('%Y-%m-%d'),
                    'predicted_price': round(max(price, 0.01), 4)
                })

            return forecast

        except Exception as e:
            logger.error(f"XGBoost prediction failed: {e}")
            return None


class HybridPredictor:
    """
    Hybrid predictor combining LSTM + XGBoost.
    - lstm_weight controls fusion: final = lstm_weight * lstm_pred + (1 - lstm_weight) * xgb_pred
    """

    def __init__(self, symbol: str, lstm_weight: float = 0.5, look_back: int = None, features: List[str] = None):
        if not 0.0 <= lstm_weight <= 1.0:
            raise ValueError("lstm_weight must be between 0 and 1")
        self.symbol = symbol
        self.lstm_weight = lstm_weight
        self.xgb_weight = 1.0 - lstm_weight
        self.look_back = look_back
        self.features = features
        self.lstm = PricePredictor(symbol, look_back=look_back, features=features)
        self.xgb = XGBoostPredictor(symbol, look_back=look_back, features=features)

    def train_or_load(self, df: pd.DataFrame, force_retrain: bool = False):
        """Train or load both models. If one fails, attempt to continue with the other."""
        logger.info(f"Hybrid: train_or_load for {self.symbol.upper()}")
        # Train/load LSTM
        try:
            self.lstm.train_or_load(df, force_retrain=force_retrain)
        except Exception as e:
            logger.warning(f"Hybrid: LSTM train_or_load failed: {e}")

        # Train/load XGBoost
        try:
            self.xgb.train_or_load(df, force_retrain=force_retrain)
        except Exception as e:
            logger.warning(f"Hybrid: XGBoost train_or_load failed: {e}")

        # Ensure at least one model is available
        if (self.lstm.model is None) and (self.xgb.model is None):
            raise RuntimeError("Hybrid training failed: no usable model available")

    def predict_future(self, df: pd.DataFrame, days: int = None) -> Optional[List[Dict]]:
        """Get forecasts from both models and fuse them"""
        days = days or Config.FORECAST_DAYS

        # Get forecasts
        lstm_forecast = None
        xgb_forecast = None

        try:
            lstm_forecast = self.lstm.predict_future(df, days=days) if self.lstm.model is not None else None
        except Exception as e:
            logger.warning(f"Hybrid: LSTM predict_future failed: {e}")
            lstm_forecast = None

        try:
            xgb_forecast = self.xgb.predict_future(df, days=days) if self.xgb.model is not None else None
        except Exception as e:
            logger.warning(f"Hybrid: XGBoost predict_future failed: {e}")
            xgb_forecast = None

        # If both failed
        if not lstm_forecast and not xgb_forecast:
            logger.error("Hybrid: Both LSTM and XGBoost failed to produce forecasts")
            return None

        # If only one is available, return that
        if lstm_forecast and not xgb_forecast:
            return lstm_forecast
        if xgb_forecast and not lstm_forecast:
            return xgb_forecast

        # Both forecasts available: fuse them (align by date)
        fused = []
        for i in range(min(len(lstm_forecast), len(xgb_forecast), days)):
            date = lstm_forecast[i]['date']
            lstm_price = float(lstm_forecast[i]['predicted_price'])
            xgb_price = float(xgb_forecast[i]['predicted_price'])
            fused_price = (self.lstm_weight * lstm_price) + (self.xgb_weight * xgb_price)
            fused.append({
                'date': date,
                'predicted_price': round(max(fused_price, 0.01), 4)
            })

        # If one forecast had more days, append remaining using available model
        max_len = max(len(lstm_forecast), len(xgb_forecast))
        if len(fused) < days and max_len > len(fused):
            # prefer LSTM remaining if available otherwise XGB
            for j in range(len(fused), days):
                if j < len(lstm_forecast):
                    fused.append(lstm_forecast[j])
                elif j < len(xgb_forecast):
                    fused.append(xgb_forecast[j])
                else:
                    break

        return fused


def get_hybrid_predictor(symbol: str, lstm_weight: float = 0.5, look_back: int = None, features: List[str] = None) -> HybridPredictor:
    """
    Factory to obtain a hybrid predictor with sensible defaults.
    - lstm_weight: weight for LSTM in fusion (0..1)
    """
    return HybridPredictor(symbol=symbol, lstm_weight=lstm_weight, look_back=look_back, features=features)


class PredictionAdjuster:
    """Adjusts raw forecast based on sentiment and technical signals (kept unchanged)"""

    def adjust(self, raw_forecast: List[Dict], decision_data: Dict,
               confidence_data: Dict, current_price: float) -> List[Dict]:
        """Adjusts the forecast based on overall signal strength and confidence"""
        if not raw_forecast:
            return []

        signal_strength = decision_data.get('signal_strength', 0) / 100.0
        confidence = confidence_data.get('overall_confidence', 50) / 100.0

        # Create a daily adjustment factor
        adjustment_factor = signal_strength * confidence * 0.5

        adjusted_forecast = []
        last_price = current_price

        for i, forecast_point in enumerate(raw_forecast):
            raw_predicted_price = forecast_point['predicted_price']

            # Calculate the model's expected change
            model_change = raw_predicted_price - last_price

            # Apply the adjustment to the change
            adjusted_change = model_change * (1 + adjustment_factor)

            # Calculate the new adjusted price
            adjusted_price = last_price + adjusted_change

            adjusted_forecast.append({
                'date': forecast_point['date'],
                'predicted_price': round(max(adjusted_price, 0.01), 2)
            })

            # The next prediction is based on the newly adjusted price
            last_price = adjusted_price

        return adjusted_forecast
