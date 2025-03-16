import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Women's Employment Prediction Model",
    page_icon="👩‍💼",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .insight-box {
        background-color: #f0f8ff;
        border-left: 5px solid #1E88E5;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .data-box {
        background-color: #f9f9f9;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .highlight {
        color: #1E88E5;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data(file_path='Data/Womens Employment.xlsx'):
    """
    Load and preprocess the women's employment dataset
    """
    try:
        # Load the dataset
        df = pd.read_excel(file_path)
        
        # Extract years from column names
        years = [str(year) for year in range(1991, 2024)]
        id_vars = [col for col in df.columns if col not in years]
        
        # Convert to long format
        df_long = pd.melt(
            df,
            id_vars=id_vars,
            value_vars=years,
            var_name='Year',
            value_name='Employment_Rate'
        )
        
        # Convert Year to integer
        df_long['Year'] = df_long['Year'].astype(int)
        
        # Handle missing values
        df_long = df_long.dropna(subset=['Employment_Rate'])
        
        # Add region mapping
        region_mapping = {
            'ARB': 'Arab World',
            'MEA': 'Middle East & North Africa',
            'MNA': 'Middle East & North Africa',
            'SAS': 'South Asia',
            'SSF': 'Sub-Saharan Africa',
            'SSA': 'Sub-Saharan Africa',
            'ECS': 'Europe & Central Asia',
            'ECA': 'Europe & Central Asia',
            'LCN': 'Latin America & Caribbean',
            'LAC': 'Latin America & Caribbean',
            'EAS': 'East Asia & Pacific',
            'EAP': 'East Asia & Pacific',
            'NAC': 'North America'
        }
        
        # Apply region mapping
        result_mapping = {}
        for code in df_long['Country Code'].unique():
            result_mapping[code] = region_mapping.get(code, 'Other')
        
        df_long['Region'] = df_long['Country Code'].map(result_mapping)
        
        # Clean data: ensure employment rates are within valid range (0-100%)
        df_long['Employment_Rate'] = df_long['Employment_Rate'].clip(0, 100)
        
        # Add flag for Arab countries
        df_long['Is_Arab'] = df_long['Region'].apply(
            lambda x: 1 if x in ['Arab World', 'Middle East & North Africa'] else 0
        ).astype(int)
        
        return df_long
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def get_clean_historical_data(df, region=None, country_code=None, min_year=2000):
    """
    Get clean historical data for a region or country
    """
    # Filter data
    filtered_df = df.copy()
    
    if region and region != "All Regions":
        filtered_df = filtered_df[filtered_df['Region'] == region]
    
    if country_code:
        filtered_df = filtered_df[filtered_df['Country Code'] == country_code]
    
    # Remove outliers within each country
    clean_data = []
    for code, country_data in filtered_df.groupby('Country Code'):
        # Skip if less than 5 data points
        if len(country_data) < 5:
            continue
        
        # Calculate metrics for outlier detection
        median = country_data['Employment_Rate'].median()
        q1 = country_data['Employment_Rate'].quantile(0.25)
        q3 = country_data['Employment_Rate'].quantile(0.75)
        iqr = q3 - q1
        lower_bound = max(0, q1 - 1.5 * iqr)
        upper_bound = min(100, q3 + 1.5 * iqr)
        
        # Filter out outliers
        valid_data = country_data[
            (country_data['Employment_Rate'] >= lower_bound) & 
            (country_data['Employment_Rate'] <= upper_bound)
        ]
        
        clean_data.append(valid_data)
    
    if not clean_data:
        return pd.DataFrame()
    
    clean_df = pd.concat(clean_data)
    
    # Filter for recent years
    clean_df = clean_df[clean_df['Year'] >= min_year]
    
    return clean_df

def aggregate_by_region(df, region=None):
    """
    Aggregate data by region and year
    """
    # Filter by region if specified
    filtered_df = df.copy()
    if region and region != "All Regions":
        filtered_df = filtered_df[filtered_df['Region'] == region]
    
    # Group by year and calculate mean employment rate
    yearly_rates = filtered_df.groupby('Year')['Employment_Rate'].agg(
        ['mean', 'std', 'count', 'min', 'max']
    ).reset_index()
    
    yearly_rates.columns = ['Year', 'Employment_Rate', 'Std_Dev', 'Count', 'Min_Rate', 'Max_Rate']
    
    return yearly_rates

def prepare_features(df):
    """
    Prepare time series features for prediction
    """
    # Make sure data is sorted by year
    df = df.sort_values('Year')
    
    # Create lag features
    df['lag_1'] = df['Employment_Rate'].shift(1)
    df['lag_2'] = df['Employment_Rate'].shift(2)
    df['lag_3'] = df['Employment_Rate'].shift(3)
    
    # Create rolling statistics
    df['rolling_mean_3'] = df['Employment_Rate'].rolling(window=3).mean().shift(1)
    df['rolling_std_3'] = df['Employment_Rate'].rolling(window=3).std().shift(1)
    
    # Calculate rate of change
    df['rate_change_1'] = df['Employment_Rate'].pct_change(1)
    df['rate_change_3'] = df['Employment_Rate'].pct_change(3)
    
    # Fixed trend feature calculation using numpy directly
    def trend_feature(series, window=5):
        if isinstance(series, pd.Series):
            # If pandas Series
            values = series.values
        else:
            # If numpy array
            values = series
            
        if len(values) >= window:
            y = values[-window:]
            x = np.arange(window)
            # Use first coefficient only, avoid full=True
            slope = np.polyfit(x, y, 1)[0]
            return slope
        return np.nan
    
    # Calculate trend for rolling window, without using raw=True
    trends = []
    for i in range(len(df)):
        if i < 4:  # Not enough data for first 4 rows
            trends.append(np.nan)
        else:
            window_data = df['Employment_Rate'].iloc[i-4:i+1]
            trends.append(trend_feature(window_data))
    
    df['trend_5'] = trends
    
    # Drop rows with NaN from feature creation
    df = df.dropna()
    
    return df

def evaluate_models(df, target_col='Employment_Rate', test_size=5):
    """
    Evaluate different models using time series cross-validation
    """
    # Prepare features
    feature_df = prepare_features(df)
    
    if len(feature_df) <= test_size:
        return None, None, "Not enough data for model evaluation"
    
    # Split into train and test
    train = feature_df.iloc[:-test_size]
    test = feature_df.iloc[-test_size:]
    
    # Define feature columns
    feature_cols = [col for col in feature_df.columns if col.startswith(('lag_', 'rolling_', 'rate_', 'trend_'))]
    
    # Define models to evaluate
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    }
    
    # Train and evaluate models
    results = {}
    feature_importances = {}
    
    for name, model in models.items():
        # Train model
        model.fit(train[feature_cols], train[target_col])
        
        # Make predictions
        train_preds = model.predict(train[feature_cols])
        test_preds = model.predict(test[feature_cols])
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(train[target_col], train_preds))
        test_rmse = np.sqrt(mean_squared_error(test[target_col], test_preds))
        test_mae = mean_absolute_error(test[target_col], test_preds)
        test_r2 = r2_score(test[target_col], test_preds)
        
        # Store results
        results[name] = {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'test_r2': test_r2,
            'model': model
        }
        
        # Get feature importances for tree-based models
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importances[name] = dict(zip(feature_cols, importances))
    
    # Find best model based on test RMSE
    best_model_name = min(results, key=lambda m: results[m]['test_rmse'])
    
    return results, feature_importances, best_model_name

def predict_with_model(historical_data, model_name, model_results, years_to_predict=5):
    """
    Make predictions using the selected model
    """
    if historical_data.empty or not model_results:
        return pd.DataFrame(), pd.DataFrame()
    
    # Aggregate by year
    yearly_data = aggregate_by_region(historical_data)
    
    # Feature preparation
    feature_df = prepare_features(yearly_data)
    
    # Get the model
    model = model_results[model_name]['model']
    
    # Get feature columns
    feature_cols = [col for col in feature_df.columns if col.startswith(('lag_', 'rolling_', 'rate_', 'trend_'))]
    
    # Create future dataframe
    future_years = list(range(yearly_data['Year'].max() + 1, 
                             yearly_data['Year'].max() + 1 + years_to_predict))
    future_df = pd.DataFrame({'Year': future_years})
    future_df['Employment_Rate'] = np.nan
    
    # Last known values for building features
    last_known_values = feature_df.iloc[-1].copy()
    
    # Iteratively predict each future year
    predictions = []
    
    for i, year in enumerate(future_years):
        # Create features for this year
        future_row = pd.Series({'Year': year})
        
        # Set lag features
        future_row['lag_1'] = last_known_values['Employment_Rate'] if i == 0 else predictions[-1]
        future_row['lag_2'] = last_known_values['lag_1'] if i == 0 else (
            last_known_values['Employment_Rate'] if i == 1 else predictions[-2]
        )
        future_row['lag_3'] = last_known_values['lag_2'] if i == 0 else (
            last_known_values['lag_1'] if i == 1 else (
                last_known_values['Employment_Rate'] if i == 2 else predictions[-3]
            )
        )
        
        # Set rolling features based on available data
        if i >= 2:  # Only calculate rolling stats when we have at least 3 points
            prev_values = np.array(predictions[-3:] if i >= 3 else 
                                ([last_known_values['Employment_Rate']] + predictions[:(i)]))
            future_row['rolling_mean_3'] = np.mean(prev_values)
            future_row['rolling_std_3'] = np.std(prev_values)
        else:
            future_row['rolling_mean_3'] = last_known_values['rolling_mean_3']
            future_row['rolling_std_3'] = last_known_values['rolling_std_3']
        
        # Set rate of change
        if i == 0:
            future_row['rate_change_1'] = last_known_values['rate_change_1']
            future_row['rate_change_3'] = last_known_values['rate_change_3']
        else:
            prev_value = last_known_values['Employment_Rate'] if i == 1 else predictions[-1]
            future_row['rate_change_1'] = (predictions[-1] - prev_value) / prev_value if prev_value != 0 else 0
            
            if i >= 3:
                prev_value_3 = last_known_values['lag_2'] if i == 3 else predictions[-3]
                future_row['rate_change_3'] = (predictions[-1] - prev_value_3) / prev_value_3 if prev_value_3 != 0 else 0
            else:
                future_row['rate_change_3'] = last_known_values['rate_change_3']
        
        # Set trend feature
        if i >= 4:
            prev_values_trend = np.array(predictions[-5:] if i >= 5 else 
                                      ([last_known_values['Employment_Rate']] + predictions[:(i)]))
            x = np.arange(len(prev_values_trend))
            slope = np.polyfit(x, prev_values_trend, 1)[0]
            future_row['trend_5'] = slope
        else:
            future_row['trend_5'] = last_known_values['trend_5']
        
        # Select only needed features
        prediction_features = {col: future_row[col] for col in feature_cols if col in future_row}
        
        # Convert to DataFrame for prediction
        pred_df = pd.DataFrame([prediction_features])
        
        # Make prediction
        try:
            pred = model.predict(pred_df)
            predicted_value = pred[0]
            
            # Ensure prediction is within reasonable bounds
            predicted_value = max(0, min(100, predicted_value))
            
            # Add to predictions list
            predictions.append(predicted_value)
            
            # Update the future dataframe
            future_df.loc[future_df['Year'] == year, 'Employment_Rate'] = predicted_value
            
        except Exception as e:
            st.error(f"Prediction error for year {year}: {str(e)}")
            break
    
    # Create predictions DataFrame
    predictions_df = future_df.rename(columns={'Employment_Rate': 'Predicted_Rate'})
    
    return predictions_df, yearly_data

def display_trend_predictions(region, historical_df):
    """
    Display predictions using advanced modeling approach
    """
    # Get clean historical data
    clean_historical = get_clean_historical_data(
        historical_df, 
        region=region if region != "All Regions" else None
    )
    
    if clean_historical.empty:
        st.error(f"Insufficient clean data for {region}")
        return
    
    # Evaluate models
    with st.spinner("Evaluating prediction models..."):
        yearly_data = aggregate_by_region(clean_historical, 
                                      region=region if region != "All Regions" else None)
        
        model_results, feature_importance, best_model = evaluate_models(yearly_data)
    
    if not model_results:
        st.error("Could not evaluate models with available data")
        return
    
    # Display model evaluation
    st.markdown(f"<h3>Model Evaluation for {region}</h3>", unsafe_allow_html=True)
    
    # Create metrics table
    metrics_df = pd.DataFrame({
        'Model': list(model_results.keys()),
        'Test RMSE': [model_results[m]['test_rmse'] for m in model_results.keys()],
        'Test MAE': [model_results[m]['test_mae'] for m in model_results.keys()],
        'R²': [model_results[m]['test_r2'] for m in model_results.keys()]
    })
    
    metrics_df = metrics_df.sort_values('Test RMSE')
    
    # Highlight the best model - compatible with older pandas versions
    best_model_idx = metrics_df[metrics_df['Model'] == best_model].index
    
    # Create a list of row styles
    row_styles = [''] * len(metrics_df)
    for idx in best_model_idx:
        row_styles[idx] = 'background-color: #e6f7ff'
    
    # Apply styling with compatible method
    def highlight_best(x):
        return [row_styles[i] for i in range(len(row_styles))]
    
    styled_df = metrics_df.style.apply(highlight_best, axis=0)
    st.dataframe(styled_df)
    
    # Add model selection explanation
    with st.expander("How is the best model selected?"):
        st.markdown(f"""
        ### Model Selection Process
        
        The system automatically evaluates multiple machine learning models and selects the best one based on their predictive performance. Here's how it works:
        
        1. **Data Preparation**: Historical employment data is processed and transformed into features that can predict future trends.
        
        2. **Model Training**: Each model is trained on the majority of historical data.
        
        3. **Model Validation**: Each model makes predictions on a validation set (most recent years that weren't used in training).
        
        4. **Performance Metrics**:
           - **RMSE (Root Mean Square Error)**: Measures the average magnitude of prediction errors. Lower values indicate better accuracy.
           - **MAE (Mean Absolute Error)**: Measures the average absolute difference between predictions and actual values. More robust to outliers than RMSE.
           - **R² (R-squared)**: Indicates how well the model explains the variance in the data. Values closer to 1 are better.
        
        5. **Selection Criteria**: The model with the lowest **Test RMSE** is selected as the best model for making future predictions. This is because RMSE penalizes larger errors more heavily, making it a good metric for selecting models that avoid significant prediction errors.
        
        In this case, **{best_model}** was selected because it achieved the lowest prediction error on recent historical data, suggesting it will likely perform best for future predictions.
        """)
        
        # Visualize model comparison
        fig_comparison = px.bar(
            metrics_df,
            x='Model',
            y='Test RMSE',
            title="Model Comparison - Lower RMSE is Better",
            color='Model',
            color_discrete_map={best_model: '#1E88E5'},
            template="plotly_white"
        )
        st.plotly_chart(fig_comparison)
    
    # Generate predictions with best model
    st.markdown(f"<h3>Prediction using {best_model}</h3>", unsafe_allow_html=True)
    
    # Generate predictions
    with st.spinner(f"Generating predictions with {best_model}..."):
        predictions_df, historical_agg = predict_with_model(
            clean_historical, best_model, model_results
        )
    
    if predictions_df.empty:
        st.error("Could not generate predictions")
        return
    
    # Show data statistics
    col1, col2, col3, col4 = st.columns(4)
    
    # For recent years only
    recent_data = historical_agg[historical_agg['Year'] >= 2010]
    
    col1.metric("Historical Mean", f"{recent_data['Employment_Rate'].mean():.2f}%")
    col2.metric("Latest Value", f"{historical_agg.iloc[-1]['Employment_Rate']:.2f}%")
    col3.metric("Min (since 2010)", f"{recent_data['Employment_Rate'].min():.2f}%")
    col4.metric("Max (since 2010)", f"{recent_data['Employment_Rate'].max():.2f}%")
    
    # Combine historical and predicted for visualization
    combined_df = pd.DataFrame({
        'Year': historical_agg['Year'].tolist() + predictions_df['Year'].tolist(),
        'Rate': historical_agg['Employment_Rate'].tolist() + predictions_df['Predicted_Rate'].tolist(),
        'Type': ['Historical'] * len(historical_agg) + ['Predicted'] * len(predictions_df)
    })
    
    # Create visualization
    fig = px.line(
        combined_df,
        x='Year',
        y='Rate',
        color='Type',
        title=f"Women's Employment Rate Prediction for {region} (2024-2028)",
        labels={"Rate": "Employment Rate (%)", "Year": "Year"},
        template="plotly_white",
        color_discrete_map={
            "Historical": "#1E88E5", 
            "Predicted": "#FFC107"
        }
    )
    
    fig.update_traces(mode='lines+markers')
    fig.update_layout(hovermode="x unified")
    
    # Add confidence interval
    years = predictions_df['Year'].values
    mean_values = predictions_df['Predicted_Rate'].values
    
    # Calculate confidence interval width based on model uncertainty
    prediction_error = model_results[best_model]['test_rmse']
    
    # Confidence interval that increases with time
    upper_bound = mean_values + [prediction_error * (1 + 0.3 * i) for i in range(len(years))]
    lower_bound = mean_values - [prediction_error * (1 + 0.3 * i) for i in range(len(years))]
    
    # Ensure bounds don't exceed 0-100%
    lower_bound = [max(0, val) for val in lower_bound]
    upper_bound = [min(100, val) for val in upper_bound]
    
    fig.add_trace(
        go.Scatter(
            x=list(years) + list(years)[::-1],
            y=list(upper_bound) + list(lower_bound)[::-1],
            fill='toself',
            fillcolor='rgba(255,193,7,0.2)',
            line=dict(color='rgba(255,193,7,0)'),
            hoverinfo='skip',
            showlegend=False,
            name='Confidence Interval'
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show predicted values
    st.markdown("<h3>Predicted Values</h3>", unsafe_allow_html=True)
    
    # Format prediction table
    prediction_table = predictions_df[['Year', 'Predicted_Rate']].copy()
    prediction_table['Predicted_Rate'] = prediction_table['Predicted_Rate'].apply(lambda x: f"{x:.2f}%")
    st.dataframe(prediction_table, hide_index=True)
    
    # Display feature importance if available
    if feature_importance and best_model in feature_importance:
        st.markdown(f"<h3>Feature Importance for {best_model}</h3>", unsafe_allow_html=True)
        
        importance_df = pd.DataFrame({
            'Feature': list(feature_importance[best_model].keys()),
            'Importance': list(feature_importance[best_model].values())
        }).sort_values('Importance', ascending=False)
        
        # Visualize feature importance
        fig_importance = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title=f"Feature Importance for {best_model}",
            template="plotly_white"
        )
        
        st.plotly_chart(fig_importance, use_container_width=True)
    
    # Calculate insights
    if not historical_agg.empty and not predictions_df.empty:
        # Get values for comparison
        latest_year = historical_agg['Year'].max()
        latest_value = historical_agg[historical_agg['Year'] == latest_year]['Employment_Rate'].values[0]
        
        final_year = predictions_df['Year'].max()
        final_value = predictions_df[predictions_df['Year'] == final_year]['Predicted_Rate'].values[0]
        
        # Calculate change
        change = final_value - latest_value
        
        # Calculate percentage change
        if latest_value > 0:
            pct_change = (change / latest_value) * 100
        else:
            pct_change = 0
        
        # Text for insights
        change_text = 'increase' if change > 0 else 'decrease'
        pct_text = 'increase' if pct_change > 0 else 'decrease'
        trend_text = 'positive' if change > 0 else 'negative'
        outcome_text = 'continuing improvement' if change > 0 else 'areas for policy focus'
        
        # Display insights
        st.markdown(
            f"""
            <div class='insight-box'>
                <strong>Prediction Insights:</strong><br>
                • Based on {best_model} analysis of historical trends, women's employment in {region} is projected to 
                <span class='highlight'>{change_text} by {abs(change):.2f} percentage points</span> 
                from {latest_year} ({latest_value:.2f}%) to {final_year} ({final_value:.2f}%).<br>
                • This represents a <span class='highlight'>{abs(pct_change):.2f}%</span> {pct_text}
                over the five-year period.<br>
                • The model achieved a prediction accuracy of R² = {model_results[best_model]['test_r2']:.3f} and RMSE = {model_results[best_model]['test_rmse']:.3f} in validation tests.<br>
                • The {trend_text} trend suggests {outcome_text} in women's economic participation in this region.
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # Detailed data table
        with st.expander("Historical Data Details"):
            st.markdown(f"<div class='data-box'>This prediction is based on data from {len(clean_historical)} observations across {clean_historical['Country Code'].nunique()} countries.</div>", unsafe_allow_html=True)
            
            # Show years with data
            years_with_data = clean_historical['Year'].value_counts().sort_index()
            st.markdown("##### Years with data:")
            st.bar_chart(years_with_data)
            
            # Show latest values by country
            latest_by_country = clean_historical.sort_values('Year').groupby('Country Code').last().reset_index()
            latest_by_country = latest_by_country.sort_values('Employment_Rate', ascending=False)
            
            st.markdown("##### Latest values by country:")
            country_data = latest_by_country[['Country Name', 'Year', 'Employment_Rate']].head(10)
            country_data['Employment_Rate'] = country_data['Employment_Rate'].apply(lambda x: f"{x:.2f}%")
            st.dataframe(country_data, hide_index=True)

def main():
    """Main function to run the app"""
    st.markdown("<h1 class='main-header'>Enhanced Women's Employment Prediction Model</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Using machine learning for accurate and reliable predictions</p>", unsafe_allow_html=True)
    
    # Add model overview at the top
    with st.expander("About This Predictive Model", expanded=True):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### How This Model Works
            
            This dashboard uses machine learning to predict future women's employment rates based on historical trends. Unlike simple linear extrapolation, our approach:
            
            1. **Evaluates Multiple Models**: The system tests Linear Regression, Ridge Regression, and Random Forest models to find the most accurate predictor.
            
            2. **Uses Advanced Features**: Instead of just extending a trend line, the model analyzes patterns like:
               - Recent employment rates
               - Rate of change over time
               - Acceleration/deceleration of trends
               - Statistical patterns in historical data
            
            3. **Selects the Best Model**: The model with the lowest prediction error on recent historical data is automatically selected to make future predictions.
            
            4. **Provides Confidence Intervals**: Predictions include uncertainty ranges that widen with time, acknowledging that longer-term predictions are less certain.
            
            This approach provides more reliable predictions for policy planning and analysis.
            """)
        
        with col2:
            st.markdown("""
            ### Key Benefits
            
            - **Higher Accuracy**: Machine learning captures complex patterns
            
            - **Model Comparison**: Automatically selects the best prediction method
            
            - **Feature Importance**: Reveals what drives employment trends
            
            - **Uncertainty Quantification**: Shows confidence intervals around predictions
            
            - **Data-Driven**: Adapts to different regional patterns
            """)

    
    # Load data
    st.subheader("1. Data Loading")
    
    # Try multiple possible locations to find the dataset
    possible_paths = [
        'Womens Employment.xlsx',
        os.path.join('data', 'Womens Employment.xlsx'),
        'Womens_Employment.xlsx',
        os.path.join('..', 'Womens Employment.xlsx'),
        os.path.join('..', 'data', 'Womens Employment.xlsx')
    ]
    
    df = None
    
    for path in possible_paths:
        if os.path.exists(path):
            st.success(f"Found dataset at: {path}")
            with st.spinner("Loading data..."):
                df = load_data(path)
            if df is not None:
                break
    
    # If no file found, allow upload
    if df is None:
        st.warning("Dataset not found. Please upload the Excel file.")
        uploaded_file = st.file_uploader("Upload Women's Employment Dataset (Excel format)", type=['xlsx', 'xls'])
        if uploaded_file is not None:
            # Save uploaded file temporarily
            temp_path = "temp_dataset.xlsx"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success("File uploaded successfully!")
            
            # Load the uploaded file
            with st.spinner("Processing data..."):
                df = load_data(temp_path)
    
    # Continue if data is loaded
    if df is not None:
        # Display basic info
        st.markdown(f"Dataset loaded with {df.shape[0]} rows and {df['Country Code'].nunique()} countries/regions")
        
        # Prediction section
        st.subheader("2. Generate Predictions")
        
        # Region selection
        regions = ["All Regions"] + sorted(df['Region'].unique().tolist())
        selected_region = st.selectbox("Select Region for Prediction", regions)
        
        # Generate predictions button
        if st.button("Generate Predictions"):
            with st.spinner("Analyzing data and generating predictions..."):
                display_trend_predictions(selected_region, df)
    else:
        st.error("Unable to process data. Please check the dataset format.")

if __name__ == "__main__":
    main()