import os
import sys
import pickle
import subprocess
import streamlit as st
import plotly.graph_objects as go
from fastapi.responses import FileResponse

def install_and_run():
    # Function to install missing packages
    def install_package(package_name):
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        except subprocess.CalledProcessError:
            print(f"Failed to install {package_name}")  
    required_packages = [
        "xgboost", "uvicorn", "nest_asyncio", "fastapi", "pydantic", "sqlalchemy", "geopy","scikit-learn"]
    for package in required_packages:
        install_package(package)

#install_and_run()
import xgboost
import uvicorn
import nest_asyncio
import shap
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from sqlalchemy import Column, Integer, String, Float, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy import text
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from sklearn.preprocessing import LabelEncoder
# Apply nest_asyncio to avoid event loop issues in Jupyter Notebook
nest_asyncio.apply()
 
# Initialize FastAPI app
app = FastAPI()
 
# Database setup
DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
 
# Define Transaction Model with updated data types
class Transaction(Base):
    __tablename__ = "transactions"
    TransactionID = Column(Integer, primary_key=True, index=True, unique=True)
    TransactionAmt = Column(Float)
    TransactionDT = Column(String)
    ProductCD = Column(String)
    User_ID = Column(Integer)
    Merchant = Column(String)
    CardNumber = Column(String)
    BINNumber = Column(String)
    CardNetwork = Column(String)
    CardTier = Column(String)
    CardType = Column(String)
    PhoneNumbers = Column(String)
    User_Region = Column(String)
    Order_Region = Column(String)
    Receiver_Region = Column(String)
    Distance = Column(Float)
    Sender_email = Column(String)
    Merchant_email = Column(String)
    DeviceType = Column(String)
    DeviceInfo = Column(String)
    # E Series Features
    TransactionTimeSlot_E2 = Column(Integer)
    HourWithinSlot_E3 = Column(Integer)
    TransactionWeekday_E4 = Column(Integer)
    AvgTransactionInterval_E5 = Column(Float)
    TransactionAmountVariance_E6 = Column(Float)
    TransactionRatio_E7 = Column(Float)
    MedianTransactionAmount_E8 = Column(Float)
    AvgTransactionAmt_24Hrs_E9 = Column(Float)
    TransactionVelocity_E10 = Column(Integer)
    TimingAnomaly_E11 = Column(Integer)
    RegionAnomaly_E12 = Column(Integer)
    HourlyTransactionCount_E13 = Column(Integer)
    # D Series Features
    DaysSinceLastTransac_D2 = Column(Float)
    SameCardDaysDiff_D3 = Column(Float)
    SameAddressDaysDiff_D4 = Column(Float)
    SameReceiverEmailDaysDiff_D10 = Column(Float)
    SameDeviceTypeDaysDiff_D11 = Column(Float)
    # C Series Features
    TransactionCount_C1 = Column(Integer)
    UniqueMerchants_C4 = Column(Integer)
    SameBRegionCount_C5 = Column(Integer)
    SameDeviceCount_C6 = Column(Integer)
    UniqueBRegion_C11 = Column(Integer)
    # M Series Features
    DeviceMatching_M4 = Column(Integer)
    DeviceMismatch_M6 = Column(Integer)
    RegionMismatch_M8 = Column(Integer)
    TransactionConsistency_M9 = Column(Integer)
    EmailFraudFlag = Column(Integer)
    # isFraud
    isFraud = Column(Integer)
 
# Create database tables
Base.metadata.create_all(bind=engine)
 
# Define request model
class TransactionIn(BaseModel):
    TransactionID: int
    TransactionAmt: float
    TransactionDT: str
    ProductCD: str
    User_ID: int
    Merchant: str
    CardNumber: str
    BINNumber: str
    CardNetwork: str
    CardTier: str
    CardType: str
    PhoneNumbers: str
    User_Region: str
    Order_Region: str
    Receiver_Region: str
    Sender_email: str
    Merchant_email: str
    DeviceType: str
    DeviceInfo: str

 
@app.get("/download-db")
def download_db():
    return FileResponse("test.db", media_type="application/octet-stream", filename="test.db")

# Helper function for DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
 
bengaluru_regions = {
        'Bagalkot': (16.1805, 75.6961), 'Ballari': (15.1394, 76.9214), 'Belagavi': (15.8497, 74.4977),
        'Bangalore': (12.9716, 77.5946), 'Bidar': (17.9106, 77.5199), 'Chamarajanagar': (11.9236, 76.9456),
        'Chikkaballapur': (13.4353, 77.7315), 'Chikkamagaluru': (13.3161, 75.7720), 'Chitradurga': (14.2296, 76.3985),
        'Dakshina Kannada': (12.8703, 74.8806), 'Davanagere': (14.4644, 75.9212), 'Dharwad': (15.4589, 75.0078),
        'Gadag': (15.4298, 75.6341), 'Hassan': (13.0057, 76.1023), 'Haveri': (14.7957, 75.3998),
        'Kalaburagi': (17.3297, 76.8376), 'Kodagu': (12.4218, 75.7400), 'Kolar': (13.1367, 78.1292),
        'Koppal': (15.3459, 76.1548), 'Mandya': (12.5223, 76.8954), 'Mysuru': (12.2958, 76.6394),
        'Raichur': (16.2076, 77.3561), 'Ramanagara': (12.7111, 77.2800), 'Shivamogga': (13.9299, 75.5681),
        'Tumakuru': (13.3409, 77.1010), 'Udupi': (13.3415, 74.7401), 'Uttara Kannada': (14.9980, 74.5070),
        'Vijayapura': (16.8302, 75.7100), 'Yadgir': (16.7625, 77.1376) }
 
def calculate_engineered_features(transaction_data: dict, db: Session):
    # Convert single transaction to DataFrame
    df = pd.DataFrame([transaction_data])
    # Convert TransactionDT to datetime if it's not already
    if isinstance(df['TransactionDT'].iloc[0], str):
        df['TransactionDT'] = pd.to_datetime(df['TransactionDT'])
 
    # Get historical transactions for the user
    historical_transactions = pd.read_sql(f"SELECT * FROM transactions WHERE User_ID = {transaction_data['User_ID']}", db.bind)
    if not historical_transactions.empty:
        historical_transactions['TransactionDT'] = pd.to_datetime(historical_transactions['TransactionDT'])
        df = pd.concat([historical_transactions, df]).reset_index(drop=True)
 
    # Calculate Distance
# Fetch last transaction's Order_Region for the user
            # Fetch last transaction's Order_Region for the user
    last_transaction = db.execute(
        text("SELECT Order_Region FROM transactions WHERE User_ID = :user_id ORDER BY TransactionDT DESC LIMIT 1"),
        {"user_id": transaction_data['User_ID']}).fetchone()

    last_order_region = last_transaction[0] if last_transaction else 0

    # For new users (no last_order_region) or if the region isn't in our database, set a default distance
    if last_order_region and last_order_region in bengaluru_regions:
        df['Distance'] = df['Order_Region'].apply(lambda region: (
            0 if region == last_order_region else  # If same, return 0
            np.round(geodesic(bengaluru_regions.get(last_order_region), bengaluru_regions.get(region)).km, 2)
            if region in bengaluru_regions else 0  # Default distance if region not in database
        ))
    else:
        # For new users, set a default distance (0 or another meaningful default)
        df['Distance'] = 0  # Default for new users

 
    # E features
    df['TransactionTimeSlot_E2'] = df['TransactionDT'].apply(lambda x: (
        0 if 10 <= x.hour < 14 else
        1 if 14 <= x.hour < 18 else
        2 if 18 <= x.hour < 22 else
        3 if x.hour >= 22 or x.hour < 2 else
        4 if 2 <= x.hour < 6 else 5
    ))
    df['HourWithinSlot_E3'] = df['TransactionDT'].apply(lambda x: (
        x.hour - 10 if 10 <= x.hour < 14 else
        x.hour - 14 if 14 <= x.hour < 18 else
        x.hour - 18 if 18 <= x.hour < 22 else
        (x.hour - 22) if x.hour >= 22 else (x.hour + 2) if x.hour < 2 else
        x.hour - 2 if 2 <= x.hour < 6 else
        x.hour - 6
    ))
    df['TransactionWeekday_E4'] = df['TransactionDT'].dt.weekday
 
    #E5
    #df['AvgTransactionInterval_E5'] = df.groupby('User_ID')['TransactionDT'].diff().dt.total_seconds() / 3600
    df = df.sort_values(by=['User_ID', 'TransactionDT']).reset_index(drop=True)
    df['AvgTransactionInterval_E5'] = df.groupby('User_ID')['TransactionDT'].diff().dt.total_seconds() / 3600
    df['AvgTransactionInterval_E5'] = df['AvgTransactionInterval_E5'].fillna(-1)
    bins = [-1, 0, 3, 100, 700, 1000, 3393]
    labels = [-1, 1, 2, 3, 4, 0]
    df['AvgTransactionInterval_E5'] = pd.cut(df['AvgTransactionInterval_E5'], bins=bins, labels=labels, right=False, include_lowest=True)
    df['AvgTransactionInterval_E5'] = df['AvgTransactionInterval_E5'].astype(int)
    
    #E6
    df['TransactionAmountVariance_E6'] = df.groupby('User_ID')['TransactionAmt'].transform('std').fillna(0)
    
    #E7
    user_mean = df.groupby('User_ID')['TransactionAmt'].transform('mean')
    df['TransactionRatio_E7'] = (df['TransactionAmt'] / user_mean).fillna(-1)
    bins = [-float('inf'), 0, 2.5, float('inf')]
    labels = [-1, 0, 1]
    df['TransactionRatio_E7'] = pd.cut(df['TransactionRatio_E7'], bins=bins, labels=labels, right=True)
    df['TransactionRatio_E7'] = df['TransactionRatio_E7'].astype(int)
    
    #E8
    df['MedianTransactionAmount_E8'] = df.groupby('User_ID')['TransactionAmt'].transform('median')
    # def median_transaction_amount(user_df):
    #     if len(user_df) == 0:
    #         return pd.Series([0] * len(user_df), index=user_df.index)
    #     median_value = user_df['TransactionAmt'].median()
    #     return pd.Series([median_value] * len(user_df), index=user_df.index)
    # df['MedianTransactionAmount_E8'] = df.groupby('User_ID').apply(
    #     median_transaction_amount).reset_index(level=0, drop=True)

    #E9
    window_24h = df.groupby('User_ID', group_keys=False).apply(lambda x: x[x['TransactionDT'] >= x['TransactionDT'].max() - pd.Timedelta(hours=24)]).reset_index(drop=True)
    #df['AvgTransactionAmt_24Hrs_E9'] = df['User_ID'].map(window_24h.groupby('User_ID')['TransactionAmt'].mean())
    df['TransactionDT'] = pd.to_datetime(df['TransactionDT'])  # Ensure it's a datetime column
    df['AvgTransactionAmt_24Hrs_E9'] = (df.groupby('User_ID', group_keys=False).apply(lambda x: x.sort_values('TransactionDT')
                                                .assign(AvgTransactionAmt_24Hrs_E9=x.rolling('24H', on='TransactionDT', min_periods=1)
                                                ['TransactionAmt'].mean()))['AvgTransactionAmt_24Hrs_E9']).fillna(0)
 
 
    
    #E10
    df['TransactionVelocity_E10'] = window_24h.groupby('User_ID')['TransactionID'].transform('count')
      
    #E11
    def timing_anomaly(group):
        if group.empty:
            return pd.Series([0] * len(group), index=group.index)
        if 'HourWithinSlot_E3' not in group.columns:
            raise KeyError("Column 'HourWithinSlot_E3' is missing in user_df")
        user_hour_freq = group['HourWithinSlot_E3'].value_counts(normalize=True)
        threshold = 0.20
        return group['HourWithinSlot_E3'].map(lambda x: 1 if user_hour_freq.get(x, 0) < threshold else 0)
    def apply_timing_anomaly(df):
        result = []
        for user_id, group in df.groupby('User_ID'):
            result.append(timing_anomaly(group))
        return pd.concat(result).sort_index()
    df['TimingAnomaly_E11'] = apply_timing_anomaly(df)

    #E12
    # user_region_freq = df.groupby(['User_ID', 'Order_Region']).size().reset_index(name='count')    
    # df['RegionAnomaly_E12'] = df.apply(
    #      lambda row: 1 if row['Order_Region'] not in user_region_freq[user_region_freq['User_ID'] == row['User_ID']]['Order_Region'].values else 0, axis=1)
    def region_anomaly(user_group, speed_threshold=30):
        if len(user_group) <= 1:  
            return pd.Series([0] * len(user_group), index=user_group.index)
        user_group = user_group.sort_values(by='TransactionDT').copy()
        user_group['Coordinates'] = user_group['Order_Region'].map(bengaluru_regions)
        user_group['Prev_Coords'] = user_group['Coordinates'].shift(1)
        user_group['Prev_Time'] = user_group['TransactionDT'].shift(1)
        distances = []
        for idx, row in user_group.iterrows():
            if pd.notna(row['Coordinates']) and pd.notna(row['Prev_Coords']):
                distances.append(geodesic(row['Coordinates'], row['Prev_Coords']).km)
            else:
                distances.append(0)
        user_group['Distance'] = distances
        user_group['TimeDiff'] = (user_group['TransactionDT'] - user_group['Prev_Time']).dt.total_seconds() / 3600
        speeds = []
        for dist, time in zip(user_group['Distance'], user_group['TimeDiff']):
            if pd.notna(time) and time > 0:
                speeds.append(dist / time)
            else:
                speeds.append(0)
        user_group['Speed'] = speeds
        return (user_group['Speed'] > speed_threshold).astype(int)
    results = []
    for name, group in df.groupby('User_ID'):
        anomalies = region_anomaly(group, speed_threshold=30)
        results.append(anomalies)
    all_results = pd.concat(results)
    df['RegionAnomaly_E12'] = all_results.reindex(df.index, fill_value=0)
    
    #E13
    df['HourlyTransactionCount_E13'] = df.groupby(['User_ID', 'HourWithinSlot_E3'])['TransactionID'].transform('count').fillna(0)
 
    # D series features
    # df['DaysSinceLastTransac_D2'] = df.groupby('User_ID')['TransactionDT'].diff().dt.total_seconds() / 86400
    # df['SameCardDaysDiff_D3'] = df.groupby('CardNumber')['TransactionDT'].diff().dt.total_seconds() / 86400
    # df['SameAddressDaysDiff_D4'] = df.groupby(['User_Region', 'Order_Region'])['TransactionDT'].diff().dt.total_seconds() / 86400
    # df['SameReceiverEmailDaysDiff_D10'] = df.groupby('Merchant_email')['TransactionDT'].diff().dt.total_seconds() / 86400
    # df['SameDeviceTypeDaysDiff_D11'] = df.groupby('DeviceType')['TransactionDT'].diff().dt.total_seconds() / 86400
    # # C series features
    # df['TransactionCount_C1'] = df.groupby(['CardNumber', 'Order_Region'])['TransactionID'].transform('count')
    # df['UniqueMerchants_C4'] = df.groupby('CardNumber')['Merchant'].transform('nunique')
    # df['SameBRegionCount_C5'] = df.groupby(['User_ID', 'User_Region'])['TransactionID'].transform('count')
    # df['SameDeviceCount_C6'] = df.groupby(['User_ID', 'DeviceType'])['TransactionID'].transform('count')
    # df['UniqueBRegion_C11'] = df.groupby('User_ID')['User_Region'].transform('nunique')
    # # M series features
    # user_common_device = df.groupby('User_ID')['DeviceType'].agg(lambda x: x.mode().iloc[0] if not x.empty else None)
    # df['DeviceMatching_M4'] = df.apply(
    #     lambda row: 1 if row['DeviceType'] == user_common_device.get(row['User_ID']) else 0, axis=1)
    # df['PrevDevice'] = df.groupby('User_ID')['DeviceType'].shift(1)
    # df['DeviceMismatch_M6'] = (df['DeviceType'] != df['PrevDevice']).astype(int)
    # df['RegionMismatch_M8'] = (df['Order_Region'] != df['User_Region']).astype(int)
    # df['TransactionConsistency_M9'] = df.apply(
    #     lambda row: sum([
    #         row['DeviceMatching_M4'],
    #         1 - row['DeviceMismatch_M6'],
    #         1 - row['RegionMismatch_M8'],
    #         1 if row['TransactionAmt'] <= row['MedianTransactionAmount_E8'] * 1.5 else 0
    #     ]), axis=1
    # )

    # D series features - with NaN handling
    df['DaysSinceLastTransac_D2'] = df.groupby('User_ID')['TransactionDT'].diff().dt.total_seconds().div(86400).fillna(0).map(lambda x: f"{x:.2f}")
    df['SameCardDaysDiff_D3'] = df.groupby('CardNumber')['TransactionDT'].diff().dt.total_seconds().div(86400).fillna(0)
    df['SameAddressDaysDiff_D4'] = df.groupby(['User_Region', 'Order_Region'])['TransactionDT'].diff().dt.total_seconds().div(86400).fillna(0)
    df['SameReceiverEmailDaysDiff_D10'] = df.groupby('Merchant_email')['TransactionDT'].diff().dt.total_seconds().div(86400).fillna(0)
    df['SameDeviceTypeDaysDiff_D11'] = df.groupby('DeviceType')['TransactionDT'].diff().dt.total_seconds().div(86400).fillna(0)

    # C series features - with NaN handling
    df['TransactionCount_C1'] = df.groupby(['CardNumber', 'Order_Region'])['TransactionID'].transform('count').fillna(1).astype(int)
    df['UniqueMerchants_C4'] = df.groupby('CardNumber')['Merchant'].transform('nunique').fillna(1).astype(int)
    df['SameBRegionCount_C5'] = df.groupby(['User_ID', 'User_Region'])['TransactionID'].transform('count').fillna(1).astype(int)
    df['SameDeviceCount_C6'] = df.groupby(['User_ID', 'DeviceType'])['TransactionID'].transform('count').fillna(1).astype(int)
    df['UniqueBRegion_C11'] = df.groupby('User_ID')['User_Region'].transform('nunique').fillna(1).astype(int)

    # M series features - with improved NaN handling
    # Safe mode() function that handles empty DataFrames
    def safe_mode(series):
        if series.empty:
            return None
        mode_values = series.mode()
        return mode_values.iloc[0] if not mode_values.empty else None

    # Calculate user common device with safe handling
    user_common_device_dict = {}
    for user_id, group in df.groupby('User_ID'):
        user_common_device_dict[user_id] = safe_mode(group['DeviceType'])

    # Apply device matching with safe handling
    df['DeviceMatching_M4'] = df.apply(
        lambda row: 1 if row['DeviceType'] == user_common_device_dict.get(row['User_ID']) else 0, 
        axis=1
    )

    # Handle PrevDevice with fillna
    df['PrevDevice'] = df.groupby('User_ID')['DeviceType'].shift(1)
    df['DeviceMismatch_M6'] = df.apply(
        lambda row: 0 if pd.isna(row['PrevDevice']) else (1 if row['DeviceType'] != row['PrevDevice'] else 0),
        axis=1
    )

    # Handle RegionMismatch more safely
    df['RegionMismatch_M8'] = df.apply(
        lambda row: 0 if pd.isna(row['Order_Region']) or pd.isna(row['User_Region']) 
                    else (1 if row['Order_Region'] != row['User_Region'] else 0),
        axis=1
    )

    # TransactionConsistency with safe handling of potential NaN values
    df['TransactionConsistency_M9'] = df.apply(
        lambda row: sum([
            row['DeviceMatching_M4'] if not pd.isna(row['DeviceMatching_M4']) else 0,
            1 - (row['DeviceMismatch_M6'] if not pd.isna(row['DeviceMismatch_M6']) else 0),
            1 - (row['RegionMismatch_M8'] if not pd.isna(row['RegionMismatch_M8']) else 0),
            1 if (not pd.isna(row['TransactionAmt']) and not pd.isna(row['MedianTransactionAmount_E8']) and 
                row['TransactionAmt'] <= row['MedianTransactionAmount_E8'] * 1.5) else 0
        ]),
        axis=1
    ).fillna(0).astype(int)

    # Extract relevant data from the transaction
    card_number = transaction_data['CardNumber']
    sender_email = transaction_data['Sender_email']

    # Query distinct Sender_email addresses for this CardNumber from historical transactions
    historical_emails = db.query(Transaction.Sender_email).filter(Transaction.CardNumber == card_number).distinct().all()
    historical_emails = set(email[0] for email in historical_emails)  # Convert to set of unique emails

    # Include the current transaction’s email if it’s not already in historical data
    if sender_email not in historical_emails:
        unique_emails_count = len(historical_emails) + 1
    else:
        unique_emails_count = len(historical_emails)

    # Set the fraud flag based on the threshold
    df['EmailFraudFlag'] = 1 if unique_emails_count > 5 else 0
    
    # # Replace NaN and inf with defaults
    # df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
 
    # Return features for the current transaction
    result = {
        'Distance': float(df.iloc[-1]['Distance']),
        'TransactionTimeSlot_E2': int(df.iloc[-1]['TransactionTimeSlot_E2']),
        'HourWithinSlot_E3': int(df.iloc[-1]['HourWithinSlot_E3']),
        'TransactionWeekday_E4': int(df.iloc[-1]['TransactionWeekday_E4']),
        'AvgTransactionInterval_E5': float(df.iloc[-1]['AvgTransactionInterval_E5']),
        'TransactionAmountVariance_E6': float(df.iloc[-1]['TransactionAmountVariance_E6']),
        'TransactionRatio_E7': float(df.iloc[-1]['TransactionRatio_E7']),
        'MedianTransactionAmount_E8': float(df.iloc[-1]['MedianTransactionAmount_E8']),
        'AvgTransactionAmt_24Hrs_E9': float(df.iloc[-1]['AvgTransactionAmt_24Hrs_E9']),
        'TransactionVelocity_E10': int(df.iloc[-1]['TransactionVelocity_E10']),
        'TimingAnomaly_E11': int(df.iloc[-1]['TimingAnomaly_E11']),
        'RegionAnomaly_E12': int(df.iloc[-1]['RegionAnomaly_E12']),
        'HourlyTransactionCount_E13': int(df.iloc[-1]['HourlyTransactionCount_E13']),
        'DaysSinceLastTransac_D2': float(df.iloc[-1]['DaysSinceLastTransac_D2']),
        'SameCardDaysDiff_D3': float(df.iloc[-1]['SameCardDaysDiff_D3']),
        'SameAddressDaysDiff_D4': float(df.iloc[-1]['SameAddressDaysDiff_D4']),
        'SameReceiverEmailDaysDiff_D10': float(df.iloc[-1]['SameReceiverEmailDaysDiff_D10']),
        'SameDeviceTypeDaysDiff_D11': float(df.iloc[-1]['SameDeviceTypeDaysDiff_D11']),
        'TransactionCount_C1': int(df.iloc[-1]['TransactionCount_C1']),
        'UniqueMerchants_C4': int(df.iloc[-1]['UniqueMerchants_C4']),
        'SameBRegionCount_C5': int(df.iloc[-1]['SameBRegionCount_C5']),
        'SameDeviceCount_C6': int(df.iloc[-1]['SameDeviceCount_C6']),
        'UniqueBRegion_C11': int(df.iloc[-1]['UniqueBRegion_C11']),
        'DeviceMatching_M4': int(df.iloc[-1]['DeviceMatching_M4']),
        'DeviceMismatch_M6': int(df.iloc[-1]['DeviceMismatch_M6']),
        'RegionMismatch_M8': int(df.iloc[-1]['RegionMismatch_M8']),
        'TransactionConsistency_M9': int(df.iloc[-1]['TransactionConsistency_M9']),
        #'EmailFraudFlag': int('email_fraud_flag')
        'EmailFraudFlag': int(df.iloc[-1]['EmailFraudFlag'])
    }
    return result
 
@app.post("/transaction_fraud_check")
async def check_transaction_fraud(transaction: TransactionIn, db: Session = Depends(get_db)):
    try:
        # Step 1: Store transaction and get engineered features
        transaction_data = transaction.model_dump()
        engineered_features = calculate_engineered_features(transaction_data, db)
        transaction_data.update(engineered_features)

        # Store transaction
        db_transaction = Transaction(**transaction_data)
        db.add(db_transaction)
        db.commit()
        db.refresh(db_transaction)

        # Step 2: Prepare data for prediction
        transaction_dict = {col.name: getattr(db_transaction, col.name) for col in Transaction.__table__.columns}
        transaction_df = pd.DataFrame([transaction_dict])
        import pickle
        import streamlit as st
         
        # Load the model
        model_path = "model1.pkl"  # Since it's in the same directory as app.py
         
        try:
            with open(model_path, "rb") as file:
                model = pickle.load(file)
            st.success("Model loaded successfully!")
        except FileNotFoundError:
            st.error(f"Model file not found at: {model_path}")
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")

        # Get expected features
        expected_features = model.feature_names_in_
        column_mapping = {
                "Cardnumber": "CardNumber",
                "UserID": "User_ID",
                "BINnumber": "BINNumber",
                "Cardnetwork": "CardNetwork",
                "Cardtier": "CardTier",
                "Cardtype": "CardType",
                "Phonenumbers": "PhoneNumbers",
                "Userregion": "User_Region",
                "Orderregion": "Order_Region",
                "Receiverregion": "Receiver_Region",
                "Senderemail": "Sender_Email",
                "Merchantemail": "Merchant_Email",
                "Devicetype": "DeviceType",
                "Deviceinfo": "DeviceInfo",
            }
        
        transaction_df.rename(columns=column_mapping, inplace=True)

        # Handle categorical columns
        categorical_cols = transaction_df.select_dtypes(include=['object']).columns
        all_transactions = pd.read_sql("SELECT * FROM transactions", db.bind)

        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            all_transactions[col] = le.fit_transform(all_transactions[col])
            transaction_df[col] = le.transform(transaction_df[col])
            label_encoders[col] = le

        # Ensure all features exist
        for col in expected_features:
            if col not in transaction_df.columns:
                transaction_df[col] = 0

        transaction_df = transaction_df[expected_features]

        # Make prediction
        prediction = model.predict(transaction_df)[0]
        prediction_proba = model.predict_proba(transaction_df)[0]

        # Handle different output formats of predict_proba
        fraud_probability = prediction_proba[1] #if len(prediction_proba) > 1 else prediction_proba

        # Apply fraud threshold
        prediction = 1.00 if fraud_probability > 0.25 else 0.00

        # Update the isFraud value in the databases
        db_transaction.isFraud = int(prediction)
        db.commit()

        if prediction == 1.00:  # Only explain fraud transactions
            # Initialize SHAP explainer
            explainer = shap.Explainer(model)
            shap_values = explainer(transaction_df)

            # Extract SHAP values for the first instance
            shap_values_instance = shap_values[0].values
            feature_names = transaction_df.columns
            # Create a DataFrame with feature names and their corresponding SHAP values
            shap_df = pd.DataFrame({
                'Feature': feature_names,
                'SHAP Value': shap_values_instance})
            # Calculate absolute SHAP values
            shap_df['Absolute SHAP Value'] = shap_df['SHAP Value'].abs()
            total_abs_shap = shap_df['Absolute SHAP Value'].sum()
            shap_df['Percentage Contribution'] = (shap_df['Absolute SHAP Value'] / total_abs_shap) * 100
            shap_df['Percentage Contribution'] = shap_df['Percentage Contribution'].map(lambda x: f"{x:.2f}")

            shap_df = shap_df.sort_values(by='Percentage Contribution', ascending=False)
            top_features = shap_df[['Feature', 'Percentage Contribution']].to_dict(orient="records")
            
            return {
                "status": "success",
                "transaction_stored": True,
                "transaction_id": transaction.TransactionID,
                "Distance": engineered_features.get("Distance", 0.0),
                "fraud_detection": {
                    "is_fraud": bool(prediction),
                    "fraud_probability": (round(float(fraud_probability),5)),
                },
                "transaction_details": {
                    "Transaction": transaction.TransactionID,
                    "Amount": transaction.TransactionAmt,
                    "Datetime": transaction.TransactionDT,
                    "Merchant": transaction.Merchant,
                    "Region": transaction.Order_Region
                },
                "Top_features": top_features
            }
        else:
            # return {
            #     "status": "failure",
            #     "transaction_id": transaction.TransactionID,
            #     "is_fraud": False,
            #     "message": "Transaction is not fraudulent, no SHAP analysis needed."
            # }
            explainer = shap.Explainer(model)
            shap_values = explainer(transaction_df)

            # Extract SHAP values for the first instance
            shap_values_instance = shap_values[0].values
            feature_names = transaction_df.columns
            # Create a DataFrame with feature names and their corresponding SHAP values
            shap_df = pd.DataFrame({
                'Feature': feature_names,
                'SHAP Value': shap_values_instance})
            # Calculate absolute SHAP values
            shap_df['Absolute SHAP Value'] = shap_df['SHAP Value'].abs()
            total_abs_shap = shap_df['Absolute SHAP Value'].sum()
            shap_df['Percentage Contribution'] = (shap_df['Absolute SHAP Value'] / total_abs_shap) * 100
            shap_df['Percentage Contribution'] = shap_df['Percentage Contribution'].map(lambda x: f"{x:.2f}")

            shap_df = shap_df.sort_values(by='Percentage Contribution', ascending=False)
            top_features = shap_df[['Feature', 'Percentage Contribution']].to_dict(orient="records")
            
            return {
                "status": "success",
                "transaction_stored": True,
                "transaction_id": transaction.TransactionID,
                "Distance": engineered_features.get("Distance", 0.0),
                "fraud_detection": {
                    "is_fraud": bool(prediction),
                    "fraud_probability": (round(float(fraud_probability),5)),
                },
                "transaction_details": {
                    "Transaction": transaction.TransactionID,
                    "Amount": transaction.TransactionAmt,
                    "Datetime": transaction.TransactionDT,
                    "Merchant": transaction.Merchant,
                    "Region": transaction.Order_Region
                },
                "Top_features": top_features}

    except Exception as e:
        db.rollback()  # Rollback transaction if error occurs
        return {
            "status": "error",
            "message": str(e)
        }
if __name__ == "__main__":
    path="model1.pkl"#str(input("Please Enter your Model.pkl path"))
    uvicorn.run(app, host="127.0.0.1", port=8000)
