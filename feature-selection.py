from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE
from sklearn.decomposition import PCA, KernelPCA
from sklearn.ensemble import RandomForestClassifier

class FeatureSelector:
    """
    Advanced feature selection and dimensionality reduction framework
    """
    def __init__(self):
        self.selected_features = {}
        self.feature_scores = {}
        
    def select_features_mutual_info(self, X, y, n_features=10):
        """
        Select features using mutual information
        """
        selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names and scores
        feature_scores = pd.DataFrame({
            'feature': X.columns,
            'score': selector.scores_
        }).sort_values('score', ascending=False)
        
        selected_features = X.columns[selector.get_support()].tolist()
        
        return X_selected, feature_scores, selected_features
    
    def recursive_feature_elimination(self, X, y, n_features=10):
        """
        Perform recursive feature elimination
        """
        estimator = RandomForestClassifier(n_estimators=100, random_state=42)
        selector = RFE(estimator=estimator, n_features_to_select=n_features)
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names and ranking
        feature_ranking = pd.DataFrame({
            'feature': X.columns,
            'ranking': selector.ranking_
        }).sort_values('ranking')
        
        selected_features = X.columns[selector.support_].tolist()
        
        return X_selected, feature_ranking, selected_features
    
    def apply_pca(self, X, n_components=0.95, kernel=None):
        """
        Apply PCA or Kernel PCA
        """
        if kernel:
            pca = KernelPCA(n_components=n_components, kernel=kernel)
        else:
            pca = PCA(n_components=n_components)
            
        X_transformed = pca.fit_transform(X)
        
        # Get explained variance ratios for PCA
        if not kernel:
            explained_variance = pd.DataFrame({
                'component': [f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
                'explained_variance_ratio': pca.explained_variance_ratio_,
                'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_)
            })
            
            return X_transformed, explained_variance
        
        return X_transformed
    
    def analyze_feature_importance(self, X, y, method='random_forest'):
        """
        Analyze feature importance using different methods
        """
        if method == 'random_forest':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            importance_scores = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
        return importance_scores
    
    def stability_selection(self, X, y, n_features=10, n_iterations=100):
        """
        Perform stability selection for feature selection
        """
        feature_frequencies = pd.DataFrame(0, index=X.columns, columns=['frequency'])
        
        for _ in range(n_iterations):
            # Random subsample
            indices = np.random.choice(len(X), size=int(0.8 * len(X)), replace=False)
            X_sample, y_sample = X.iloc[indices], y.iloc[indices]
            
            # Feature selection on subsample
            selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
            selector.fit(X_sample, y_sample)
            
            # Update frequencies
            selected_features = X.columns[selector.get_support()].tolist()
            feature_frequencies.loc[selected_features, 'frequency'] += 1
            
        feature_frequencies['frequency'] = feature_frequencies['frequency'] / n_iterations
        
        return feature_frequencies.sort_values('frequency', ascending=False)