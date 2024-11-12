import scipy.stats as stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller
import numpy as np
import pandas as pd

class StatisticalAnalyzer:
    """
    Advanced statistical analysis framework for predictive maintenance data
    """
    def __init__(self):
        self.analysis_results = {}
        
    def analyze_distributions(self, df):
        """
        Comprehensive distribution analysis
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        distribution_tests = {}
        
        for col in numeric_cols:
            # Normality tests
            shapiro_test = stats.shapiro(df[col])
            ks_test = stats.kstest(df[col], 'norm')
            anderson_test = stats.anderson(df[col], dist='norm')
            
            # Skewness and Kurtosis
            skew = stats.skew(df[col])
            kurt = stats.kurtosis(df[col])
            
            distribution_tests[col] = {
                'shapiro_test': {'statistic': shapiro_test[0], 'p_value': shapiro_test[1]},
                'ks_test': {'statistic': ks_test[0], 'p_value': ks_test[1]},
                'anderson_test': {'statistic': anderson_test.statistic, 
                                'critical_values': anderson_test.critical_values},
                'skewness': skew,
                'kurtosis': kurt
            }
        
        return pd.DataFrame(distribution_tests)
    
    def check_multicollinearity(self, df):
        """
        Calculate Variance Inflation Factor (VIF) for multicollinearity
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        vif_data = pd.DataFrame()
        vif_data["Feature"] = numeric_cols
        vif_data["VIF"] = [variance_inflation_factor(df[numeric_cols].values, i)
                          for i in range(df[numeric_cols].shape[1])]
        
        return vif_data.sort_values('VIF', ascending=False)
    
    def analyze_stationarity(self, df, time_col='Tool wear'):
        """
        Perform stationarity analysis for time series aspects
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = numeric_cols.drop(time_col) if time_col in numeric_cols else numeric_cols
        
        stationarity_results = {}
        for col in numeric_cols:
            # Augmented Dickey-Fuller test
            adf_test = adfuller(df[col])
            stationarity_results[col] = {
                'adf_statistic': adf_test[0],
                'p_value': adf_test[1],
                'critical_values': adf_test[4]
            }
        
        return pd.DataFrame(stationarity_results)
    
    def perform_hypothesis_tests(self, df, target_col='Machine failure'):
        """
        Comprehensive hypothesis testing between features and target
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = numeric_cols.drop(target_col) if target_col in numeric_cols else numeric_cols
        
        hypothesis_tests = {}
        for col in numeric_cols:
            # Mann-Whitney U test
            mw_test = stats.mannwhitneyu(
                df[df[target_col] == 0][col],
                df[df[target_col] == 1][col],
                alternative='two-sided'
            )
            
            # Effect size (Cohen's d)
            cohens_d = (df[df[target_col] == 0][col].mean() - 
                       df[df[target_col] == 1][col].mean()) / (
                           np.sqrt((df[df[target_col] == 0][col].var() + 
                                  df[df[target_col] == 1][col].var()) / 2)
                       )
            
            hypothesis_tests[col] = {
                'mann_whitney_statistic': mw_test[0],
                'p_value': mw_test[1],
                'cohens_d': cohens_d
            }
        
        return pd.DataFrame(hypothesis_tests)
    
    def analyze_feature_stability(self, df, time_col='Tool wear', window_size=100):
        """
        Analyze feature stability over time
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = numeric_cols.drop(time_col) if time_col in numeric_cols else numeric_cols
        
        stability_metrics = {}
        for col in numeric_cols:
            # Calculate rolling statistics
            rolling_mean = df[col].rolling(window=window_size).mean()
            rolling_std = df[col].rolling(window=window_size).std()
            
            # Calculate stability metrics
            stability_metrics[col] = {
                'mean_stability': rolling_mean.std() / df[col].mean(),
                'std_stability': rolling_std.std() / df[col].std(),
                'range_stability': (df[col].max() - df[col].min()) / df[col].mean()
            }
        
        return pd.DataFrame(stability_metrics)