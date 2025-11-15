import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin



class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, rolling_window=3):
        self.rolling_window = rolling_window

    def fit(self, X, y=None):
        # Nothing to fit here, but return self to be consistent with sklearn API
        return self

    def transform(self, X):
        X = X.copy()

        # Price Change per Product
        if 'Product ID' in X.columns and 'Price' in X.columns:
            X['Price Change'] = X.groupby('Product ID')['Price'].diff().fillna(0)
        else:
            X['Price Change'] = 0  # fallback if missing during inference

        # Effective Price after discount
        if 'Price' in X.columns and 'Discount' in X.columns:
            X['Effective Price'] = X['Price'] * (1 - X['Discount'] / 100)
        else:
            X['Effective Price'] = 0  # fallback

        # Rolling Average of Units Sold per Store-Product
        if 'Store ID' in X.columns and 'Product ID' in X.columns and 'Units Sold' in X.columns:
            X['Units Sold Rolling Avg'] = (
                X.groupby(['Store ID', 'Product ID'])['Units Sold']
                 .transform(lambda x: x.rolling(window=self.rolling_window, min_periods=1).mean())
            )
        else:
            X['Units Sold Rolling Avg'] = 0

        # Extract Date features if Date is present
        if 'Date' in X.columns:
            X['Year'] = X['Date'].dt.year
            X['Month'] = X['Date'].dt.month
            X.drop(['Date'], axis=1, inplace=True)
        else:
            X['Year'] = 0
            X['Month'] = 0

        return X