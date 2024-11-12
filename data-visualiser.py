import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

class DataVisualizer:
    """
    Advanced visualization framework for predictive maintenance data
    """
    def __init__(self, style='whitegrid'):
        sns.set_style(style)
        self.colors = sns.color_palette("husl", 8)
        
    def plot_feature_distributions(self, df, target_col='Machine failure'):
        """
        Plot distribution of features with respect to target variable
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = numeric_cols.drop(target_col) if target_col in numeric_cols else numeric_cols
        
        n_cols = 3
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten()
        
        for idx, col in enumerate(numeric_cols):
            # Distribution plot
            sns.kdeplot(data=df, x=col, hue=target_col, ax=axes[idx])
            axes[idx].set_title(f'Distribution of {col} by {target_col}')
            
            # Add statistical test results
            stat, p_value = stats.ks_2samp(
                df[df[target_col] == 0][col],
                df[df[target_col] == 1][col]
            )
            axes[idx].text(0.05, 0.95, f'KS test p-value: {p_value:.2e}',
                         transform=axes[idx].transAxes)
            
        plt.tight_layout()
        return fig
        
    def plot_correlation_heatmap(self, df, method='pearson'):
        """
        Enhanced correlation heatmap with hierarchical clustering
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr = df[numeric_cols].corr(method=method)
        
        # Generate mask for upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr, mask=mask, cmap='coolwarm', center=0,
                   annot=True, fmt='.2f', square=True,
                   linewidths=0.5, cbar_kws={"shrink": .5})
        plt.title(f'{method.capitalize()} Correlation Heatmap')
        
        return plt.gcf()
        
    def plot_temporal_patterns(self, df, time_col='Tool wear', value_cols=None):
        """
        Analyze and plot temporal patterns in the data
        """
        if value_cols is None:
            value_cols = df.select_dtypes(include=[np.number]).columns.drop(time_col)
        
        n_cols = 2
        n_rows = (len(value_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten()
        
        for idx, col in enumerate(value_cols):
            sns.scatterplot(data=df, x=time_col, y=col, ax=axes[idx])
            
            # Add trend line
            z = np.polyfit(df[time_col], df[col], 1)
            p = np.poly1d(z)
            axes[idx].plot(df[time_col], p(df[time_col]), "r--", alpha=0.8)
            
            axes[idx].set_title(f'{col} vs {time_col}')
            
        plt.tight_layout()
        return fig
        
    def plot_feature_importance(self, df, target_col='Machine failure'):
        """
        Plot feature importance based on mutual information
        """
        from sklearn.feature_selection import mutual_info_classif
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = numeric_cols.drop(target_col) if target_col in numeric_cols else numeric_cols
        
        # Calculate mutual information scores
        mi_scores = mutual_info_classif(df[numeric_cols], df[target_col])
        
        # Create DataFrame of features and their importance scores
        feature_importance = pd.DataFrame({
            'Feature': numeric_cols,
            'Importance': mi_scores
        }).sort_values('Importance', ascending=True)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=feature_importance, y='Feature', x='Importance')
        plt.title('Feature Importance Based on Mutual Information')
        
        return plt.gcf()
        
    def plot_outlier_analysis(self, df):
        """
        Create comprehensive outlier visualization
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        n_cols = 2
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten()
        
        for idx, col in enumerate(numeric_cols):
            # Box plot
            sns.boxplot(y=df[col], ax=axes[idx])
            axes[idx].set_title(f'Outliers in {col}')
            
            # Add statistical measurements
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            outliers = df[
                (df[col] < (q1 - 1.5 * iqr)) | 
                (df[col] > (q3 + 1.5 * iqr))
            ][col]
            
            axes[idx].text(0.05, 0.95, 
                         f'Outliers: {len(outliers)}\nOutlier %: {(len(outliers)/len(df)*100):.2f}%',
                         transform=axes[idx].transAxes,
                         bbox=dict(facecolor='white', alpha=0.8))
            
        plt.tight_layout()
        return fig