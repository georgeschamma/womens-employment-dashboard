# Modify the file path handling in your main() function with this approach:

def main():
    """Main function to run the app"""
    st.markdown("<h1 class='main-header'>Enhanced Women's Employment Prediction Model</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Using machine learning for accurate and reliable predictions</p>", unsafe_allow_html=True)
    
    # Add model overview at the top
    # [Your existing expander code here]
    
    # Load data
    st.subheader("1. Data Loading")
    
    # Debug information to understand the environment
    st.sidebar.expander("Debug Info", expanded=False).write(f"""
    - Working directory: {os.getcwd()}
    - Files in directory: {os.listdir('.')}
    """)
    
    # Try to find the dataset in multiple locations with better debugging
    possible_paths = [
        'Womens Employment.xlsx',
        'data/Womens Employment.xlsx',
        'Womens_Employment.xlsx',
        '../Womens Employment.xlsx',
        '../data/Womens Employment.xlsx',
        './Womens Employment.xlsx',
        './data/Womens Employment.xlsx'
    ]
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    df = None
    data_path = None
    
    # Check each possible path
    for path in possible_paths:
        if os.path.exists(path):
            data_path = path
            st.success(f"Found dataset at: {path}")
            with st.spinner("Loading data..."):
                df = load_data(path)
            if df is not None:
                break
    
    # If no file found, allow upload
    if df is None:
        st.warning("Dataset not found. Please upload the Excel file.")
        
        # Instructions for permanent fix
        st.info("""
        **To fix this permanently:** 
        1. Upload your data file below
        2. The file will be saved to the app's storage
        3. Refresh the page after uploading - the app should then find the file automatically
        """)
        
        uploaded_file = st.file_uploader("Upload Women's Employment Dataset (Excel format)", type=['xlsx', 'xls'])
        if uploaded_file is not None:
            # Save to multiple potential locations to ensure it's found next time
            save_paths = ['Womens Employment.xlsx', 'data/Womens Employment.xlsx']
            
            for save_path in save_paths:
                # Ensure directory exists
                os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
                
                # Save the file
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            
            st.success(f"File uploaded successfully and saved to {save_paths}!")
            
            # Load the uploaded file
            with st.spinner("Processing data..."):
                df = load_data(save_paths[0])  # Use the first path for loading
    
    # Continue if data is loaded
    if df is not None:
        # Display basic info
        st.markdown(f"Dataset loaded with {df.shape[0]} rows and {df['Country Code'].nunique()} countries/regions")
        
        # [Rest of your existing code...]
