import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """
    Advanced preprocessing framework for predictive maintenance data
    """
    def __init__(self):
        self.scalers = {}
        self.outlier_thresholds = {}
        self.feature_stats = {}
        
    def analyze_data_quality(self, df):
        """
        Comprehensive data quality analysis
        """
        quality_report = {
            'missing_values': df.isnull().sum(),
            'duplicates': df.duplicated().sum(),
            'unique_counts': df.nunique(),
            'data_types': df.dtypes,
            'skewness': df.select_dtypes(include=[np.number]).skew(),
            'kurtosis': df.select_dtypes(include=[np.number]).kurtosis()
        }
        
        # Check for near-zero variance
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        variance = df[numeric_cols].var()
        quality_report['low_variance_features'] = variance[variance < 0.01].index.tolist()
        
        # Distribution tests
        normality_tests = {}
        for col in numeric_cols:
            stat, p_value = stats.normaltest(df[col])
            normality_tests[col] = {'statistic': stat, 'p_value': p_value}
        quality_report['normality_tests'] = normality_tests
        
        return pd.DataFrame(quality_report)

    def detect_outliers_advanced(self, df, methods=['zscore', 'iqr', 'isolation_forest']):
        """
        Multi-method outlier detection
        """
        outlier_report = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            outliers = set()
            
            if 'zscore' in methods:
                z_scores = np.abs(stats.zscore(df[col]))
                z_score_outliers = df.index[z_scores > 3].tolist()
                outliers.update(z_score_outliers)
            
            if 'iqr' in methods:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                iqr_outliers = df[
                    (df[col] < (Q1 - 1.5 * IQR)) | 
                    (df[col] > (Q3 + 1.5 * IQR))
                ].index.tolist()
                outliers.update(iqr_outliers)
            
            if 'isolation_forest' in methods:
                from sklearn.ensemble import IsolationForest
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                yhat = iso_forest.fit_predict(df[col].values.reshape(-1, 1))
                iso_outliers = df.index[yhat == -1].tolist()
                outliers.update(iso_outliers)
            
            outlier_report[col] = list(outliers)
            
        return outlier_report

    def engineer_features(self, df):
        """
        Advanced feature engineering specific to predictive maintenance
        """
        df_engineered = df.copy()
        
        # Create interaction terms for temperatures
        df_engineered['temp_difference'] = df['Process temperature'] - df['Air temperature']
        df_engineered['temp_ratio'] = df['Process temperature'] / df['Air temperature']
        
        # Power-related features
        df_engineered['power_normalized'] = df_engineered['Power'] / df_engineered['Rotational speed']
        df_engineered['torque_per_wear'] = df['Torque'] / (df['Tool wear'] + 1)  # Adding 1 to avoid division by zero
        
        # Moving averages and variations (if data is time-ordered)
        windows = [3, 5, 7]
        for window in windows:
            df_engineered[f'power_ma_{window}'] = df_engineered['Power'].rolling(window=window).mean()
            df_engineered[f'temp_diff_ma_{window}'] = df_engineered['temp_difference'].rolling(window=window).mean()
        
        # Polynomial features for key metrics
        df_engineered['tool_wear_squared'] = df['Tool wear'] ** 2
        df_engineered['rotational_speed_squared'] = df['Rotational speed'] ** 2
        
        return df_engineered

    def scale_features(self, df, method='robust'):
        """
        Scale features with option for different scaling methods
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        scaled_df = df.copy()
        
        for col in numeric_cols:
            if method == 'robust':
                scaler = RobustScaler()
            elif method == 'standard':
                scaler = StandardScaler()
            
            scaled_df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1))
            self.scalers[col] = scaler
            
        return scaled_df

    def reduce_dimensionality(self, df, n_components=0.95):
        """
        Perform dimensionality reduction with PCA
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(df[numeric_cols])
        
        # Create DataFrame with PCA results
        cols = [f'PC{i+1}' for i in range(pca_result.shape[1])]
        pca_df = pd.DataFrame(pca_result, columns=cols, index=df.index)
        
        # Calculate and store explained variance
        explained_variance = pd.DataFrame(
            pca.explained_variance_ratio_,
            columns=['Explained Variance'],
            index=cols
        )
        
        return pca_df, explained_variance

    def analyze_correlations(self, df, threshold=0.7):
        """
        Advanced correlation analysis
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()
        
        # Find highly correlated features
        high_correlations = np.where(np.abs(corr_matrix) > threshold)
        high_correlations = [(corr_matrix.index[x], corr_matrix.columns[y], corr_matrix.iloc[x, y])
                           for x, y in zip(*high_correlations) if x != y and x < y]
        
        return pd.DataFrame(high_correlations, columns=['Feature 1', 'Feature 2', 'Correlation'])