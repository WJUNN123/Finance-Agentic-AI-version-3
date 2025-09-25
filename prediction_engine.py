"""
LSTM-based price prediction engine
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
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

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
    """LSTM model for price prediction with model persistence"""
    
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
        # Check if required features exist in the DataFrame
        missing_features = [f for f in self.features if f not in df.columns]
        if missing_features:
            logger.warning(f"Missing features {missing_features} in DataFrame")
            # Use available features only
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
        """Main method to either load a pre-trained model or train a new one"""
        if (self.model_path.exists() and self.scaler_path.exists() 
            and not force_retrain):
            try:
                logger.info(f"Loading pre-trained model for {self.symbol.upper()}")
                self.model = tf.keras.models.load_model(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                logger.info("Model loaded successfully")
                return
            except Exception as e:
                logger.warning(f"Could not load model, will retrain. Error: {e}")

        logger.info(f"Training new LSTM model for {self.symbol.upper()}")

        # Check if we have enough data
        min_required = self.look_back + 20
        if len(df) < min_required:
            logger.warning(
                f"Not enough historical data to train optimally. "
                f"Required {min_required}, got {len(df)}"
            )
            if len(df) < self.look_back + 5:
                raise ValueError(
                    f"Insufficient data for training. "
                    f"Need at least {self.look_back + 5} rows, got {len(df)}"
                )

        _, scaled_data = self._prepare_data(df)
        X_train, y_train = self._create_dataset(scaled_data)

        if X_train.shape[0] == 0:
            raise RuntimeError("Could not create a training set from the provided data")

        logger.info(f"Preparing training data: {X_train.shape[0]} samples")

        self._build_model((X_train.shape[1], X_train.shape[2]))

        # Callbacks for training
        early_stopping = EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
        model_checkpoint = ModelCheckpoint(
            self.model_path, save_best_only=True, monitor='val_loss'
        )

        # Train the model with suppressed output
        logger.info("Training model...")
        
        with suppress_stdout():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model.fit(
                    X_train, y_train,
                    batch_size=min(32, len(X_train) // 4),
                    epochs=50,
                    validation_split=0.1,
                    callbacks=[early_stopping, model_checkpoint],
                    verbose=0
                )

        logger.info(f"Model training completed for {self.symbol.upper()}")

        # Save the scaler
        joblib.dump(self.scaler, self.scaler_path)

        # Verify model and scaler are properly loaded
        if self.model is None or self.scaler is None:
            raise RuntimeError(
                f"Model for {self.symbol} was not trained or loaded successfully"
            )

    def predict_future(self, df: pd.DataFrame, days: int = None) -> Optional[List[Dict]]:
        """Predict future prices using the trained multivariate model"""
        days = days or Config.FORECAST_DAYS
        
        if self.model is None:
            logger.error("Model is not loaded. Please call train_or_load() first")
            return None

        try:
            # Use the same features as during training
            data = df.filter(self.features)

            # Handle missing values
            data = data.ffill().bfill()
            data = data.replace([np.inf, -np.inf], np.nan).ffill()

            if len(data) < self.look_back:
                logger.error(
                    f"Not enough data for prediction. "
                    f"Need {self.look_back}, got {len(data)}"
                )
                return None

            scaled_data = self.scaler.transform(data.values)
            last_sequence = scaled_data[-self.look_back:]
            predictions = []

            logger.info(f"Generating {days}-day forecast...")

            for day in range(days):
                # Reshape for prediction
                current_sequence_reshaped = np.reshape(
                    last_sequence, (1, self.look_back, len(self.features))
                )

                # Make prediction with suppressed output
                with suppress_stdout():
                    predicted_scaled_price = self.model.predict(
                        current_sequence_reshaped, verbose=0
                    )[0][0]
                    
                predictions.append(predicted_scaled_price)

                # Update sequence for next prediction
                if len(self.features) > 1:
                    next_step_features = last_sequence[-1, 1:].copy()
                    new_row = np.insert(next_step_features, 0, predicted_scaled_price)
                else:
                    new_row = np.array([predicted_scaled_price])

                # Update the sequence
                last_sequence = np.vstack([last_sequence[1:], new_row.reshape(1, -1)])

            # Convert back to original scale
            dummy_array = np.zeros((len(predictions), len(self.features)))
            dummy_array[:, 0] = predictions
            predicted_prices_unscaled = self.scaler.inverse_transform(dummy_array)[:, 0]

            # Format forecast
            forecast = []
            for i, price in enumerate(predicted_prices_unscaled):
                forecast.append({
                    'date': (datetime.now() + timedelta(days=i + 1)).strftime('%Y-%m-%d'),
                    'predicted_price': round(max(price, 0.01), 4)
                })

            return forecast

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return None

class PredictionAdjuster:
    """Adjusts raw LSTM forecast based on sentiment and technical signals"""
    
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