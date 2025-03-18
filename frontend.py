import streamlit as st
import re
import time
from datetime import datetime
import requests
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px

def mask_card_number(card_number):
    """Masks all but the last 4 digits of the card number."""
    if card_number and len(card_number) > 4:
        return "XXXX XXXX XXXX " + card_number[-4:]
    return "XXXX XXXX XXXX"

def fraud_meter(result):
    fraud_probability = result["fraud_detection"]["fraud_probability"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=fraud_probability,
        title={'text': "Fraud Probability", 'font': {'size': 20}},
        gauge={'axis': {'range': [None, 1]},
            'bar': {'color': "black"},
            'steps': [
                {'range': [0, 0.2], 'color': "green"},
                {'range': [0.2, 0.4], 'color': "#90EE90"},
                {'range': [0.4, 0.6], 'color': "yellow"},
                {'range': [0.6, 0.8], 'color': "orange"},
                {'range': [0.8, 1], 'color': "red"}],
            'threshold': {'line': {'color': "red", 'width': 2}, 'value': fraud_probability}
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

def fraud_confidence_display(result):
    """Returns fraud confidence data to be displayed"""
    fraud_confidence = result["fraud_detection"]["fraud_confidence_score"]
    
    # Determine color, label, and icon based on confidence score
    if fraud_confidence <= 30:
        color = "#4CAF50"  # Green
        label = "Legitimate"
        icon = "👤"
    elif fraud_confidence <= 60:
        color = "#FFC107"  # Yellow
        label = "Uncertain"
        icon = "👤"
    else:
        color = "#F44336"  # Red
        label = "Synthetic"
        icon = "👤"
    
    return {
        "color": color,
        "label": label,
        "icon": icon,
        "confidence": fraud_confidence
    }

def display_top_features(result):
    try:
        top_features = result['Top_features']
        if not top_features:
            st.info("No feature importance data available for this transaction.")
            return

        features_df = pd.DataFrame(top_features)
        if 'Feature' not in features_df.columns or 'Percentage Contribution' not in features_df.columns:
            st.warning("Unexpected feature importance format.")
            st.write(top_features)
            return

        features_df['Percentage Contribution'] = pd.to_numeric(features_df['Percentage Contribution'], errors='coerce')
        features_df = features_df.sort_values(by="Percentage Contribution", ascending=False)
        features_df['Cumulative Percentage'] = features_df['Percentage Contribution'].cumsum()

        top_features = features_df[features_df['Cumulative Percentage'] <= 90]
        others = features_df[features_df['Cumulative Percentage'] > 90]
        if not others.empty:
            others_row = pd.DataFrame([{
                'Feature': 'Others',
                'Percentage Contribution': others['Percentage Contribution'].sum(),
                'Cumulative Percentage': 100
            }])
            top_features = pd.concat([top_features, others_row], ignore_index=True)

        feature_explanations = {
            "TransactionAmt": "Fraudsters often attempt high-value transactions to maximize their profit before detection.",
            "TransactionDT": "Suspicious transactions may occur at unusual hours, revealing fraudulent patterns.",
            "ProductCD": "Certain product categories are more prone to fraud, like high-value electronics or gift cards.",
            "User_ID": "Multiple User_IDs linked to the same card or region can indicate identity theft.",
            "Merchant": "Transactions with high-risk merchants or unfamiliar vendors may signal fraud.",
            "CardNumber": "Unusual or rarely used card numbers are a red flag for fraudulent activity.",
            "BINNumber": "BIN anomalies suggest cards from suspicious or blacklisted banks.",
            "CardNetwork": "Fraudulent transactions may use lesser-known or compromised card networks.",
            "CardTier": "High-tier cards (Platinum, Gold) are often targeted for their high credit limits.",
            "CardType": "Credit cards are more susceptible to fraud than debit cards due to higher limits.",
            "PhoneNumbers": "Multiple phone numbers linked to a single user can suggest account takeovers.",
            "User_Region": "Transactions from unexpected regions could indicate a compromised account.",
            "Order_Region": "A mismatch between user and order regions is a sign of potential fraud.",
            "Receiver_Region": "Unusual receiver regions may reveal cross-district fraud.",
            "Distance": "Large distances between billing and transaction locations can raise suspicion.",
            "Sender_email": "New or uncommon email domains may be tied to fraudulent activity.",
            "Merchant_email": "Merchants with unverified or mismatched emails are a risk factor.",
            "DeviceType": "Device changes (mobile vs. desktop) can indicate unauthorized access.",
            "DeviceInfo": "Using unknown or outdated devices might signal a fraud attempt.",
            "TransactionTimeSlot_E2": "Transactions in low-activity time slots may indicate automated fraud.",
            "HourWithinSlot_E3": "Irregular hours within a low-activity time slot can hint at suspicious behavior.",
            "TransactionWeekday_E4": "Fraudulent activities often spike on weekends or holidays.",
            "AvgTransactionInterval_E5": "Short transaction intervals may indicate rapid fraudulent purchases.",
            "TransactionAmountVariance_E6": "High variance in transaction amounts suggests inconsistent activity.",
            "TransactionRatio_E7": "An unusual transaction ratio may highlight fraud-prone accounts.",
            "MedianTransactionAmount_E8": "Deviations from median amounts can expose fraudulent behavior.",
            "AvgTransactionAmt24Hrs_E9": "Spikes in daily transaction amounts might reveal fraud bursts.",
            "TransactionVelocity_E10": "A high volume of quick transactions could be fraud bot behavior.",
            "TimingAnomaly_E11": "Unusual transaction times can be a strong fraud indicator.",
            "RegionAnomaly_E12": "Transactions in rare regions may uncover regional fraud schemes.",
            "HourlyTransactionCount_E13": "A sudden surge in hourly transactions often signals bot attacks.",
            "EmailFraudFlag_E14": "Email fraud indicators can help flag risky email domains.",
            "DaysSinceLastTransac_D2": "A long gap followed by sudden activity may reflect card testing fraud.",
            "SameCardDaysDiff_D3": "Quick repeat use of the same card raises fraud alarms.",
            "SameAddressDaysDiff_D4": "Repeated transactions from the same address could be fraud stacking.",
            "SameReceiverEmailDaysDiff_D10": "Multiple fast transactions to the same receiver hint at scams.",
            "SameDeviceTypeDaysDiff_D11": "A device consistently linked to fraud shows suspicious behavior.",
            "TransactionCount_C1": "Unusually high transaction counts per card may indicate abuse.",
            "UniqueMerchants_C4": "Frequent transactions across various merchants can expose fraud rings.",
            "SameBRegionCount_C5": "A high count of same billing regions might show location spoofing.",
            "SameDeviceCount_C6": "Multiple transactions from the same device can indicate a bot.",
            "UniqueBRegion_C11": "Transactions from diverse regions in quick succession suggest fraud.",
            "DeviceMatching_M4": "Matching devices indicate normal behavior; mismatches hint at fraud.",
            "DeviceMismatch_M6": "Device mismatches are a strong fraud risk factor.",
            "RegionMismatch_M8": "A billing and transaction region mismatch suggests unauthorized activity.",
            "TransactionConsistency_M9": "Inconsistent transaction patterns indicate suspicious behavior."
        }

        hover_template = """
            <b>%{label}</b><br>
            🚀 Contribution to Fraud Risk: %{value:.2f}%<br>
            📊 Reason: """ + "%{customdata[0]}" + """<br>
            🏷️ Feature: %{label}
        """

        top_features['Explanation'] = top_features['Feature'].map(feature_explanations)

        fig = px.sunburst(
            top_features,
            path=['Feature'],
            values='Percentage Contribution',
            title="Key Factors in Fraud Detection Decision",
            color='Percentage Contribution',
            color_continuous_scale='Oranges',
            custom_data=['Explanation']
        )

        fig.update_traces(
            textinfo='label+percent entry',
            insidetextfont=dict(color='black'),
            hovertemplate=hover_template
        )

        with st.expander("🔍 View Risk Factor Analysis", expanded=False):
            st.markdown("### Analysis of Risk Factors")
            st.write("The sunburst chart below visualizes key factors influencing the fraud detection decision:")
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error displaying feature importance: {str(e)}")

def transaction_page():
    st.set_page_config(page_title="Transaction Entry", page_icon='👤', layout="wide")

    st.markdown("""
    <div class="main-header">
    <div>
    <h1 class="header-title">Fraud Shield 🛡️</h1>
    <p class="header-subtitle">Analyze transaction risk with AI-powered detection 🔍💻</p>
    </div>
    </div>
    """, unsafe_allow_html=True)

    def load_css(file_path):
        try:
            with open(file_path, "r") as f:
                return f"<style>{f.read()}</style>"
        except FileNotFoundError:
            return ""

    st.markdown(load_css("styles.css"), unsafe_allow_html=True)

    if 'sidebar_open' not in st.session_state:
        st.session_state.sidebar_open = False
    if "transaction_dt" not in st.session_state:
        st.session_state.transaction_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if 'selected_category' not in st.session_state:
        st.session_state.selected_category = "Retail"
    if 'selected_product' not in st.session_state:
        st.session_state.selected_product = "Books"
    if 'transaction_amount' not in st.session_state:
        st.session_state.transaction_amount = 1500.00
    if 'selected_merchant' not in st.session_state:
        st.session_state.selected_merchant = "Flipkart"
    if 'merchant_email' not in st.session_state:
        st.session_state.merchant_email = "retail@flipkart.com"
    if 'sender_email' not in st.session_state:
        st.session_state.sender_email = "poll@gmail.com"

    def validate_email(email):
        pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
        return bool(re.match(pattern, email))

    def validate_card_number(card_number):
        return bool(isinstance(card_number, (int, str)) and str(card_number).isdigit() and len(str(card_number)) == 16)

    def validate_bin_number(bin_number):
        return bool(isinstance(bin_number, (int, str)) and str(bin_number).isdigit() and len(str(bin_number)) == 6)

    def extract_bin(card_number):
        if validate_card_number(card_number):
            return str(card_number)[:6]
        return ""

    def send_to_backend(transaction_data):
        try:
            response = requests.post("http://127.0.0.1:8000/transaction_fraud_check", json=transaction_data)
            response.raise_for_status()
            result = response.json()
            return True, result
        except requests.exceptions.RequestException as e:
            return False, f"Error: {str(e)}"

    default_amounts = {
        "Shoes": 1500.00,
        "Smartphone": 15000.00,
        "Jewelry": 20000.00,
        "Beauty Products": 1000.00,
        "Books": 500.00,
        "PhonePe Wallet": 1000.00,
        "Paytm Wallet": 1000.00,
        "Google Pay Wallet": 1000.00,
        "Amazon Pay Wallet": 1000.00,
        "Cleaning Products": 500.00,
        "Personal Care Products": 800.00,
        "Health Supplements": 2000.00,
        "Fruits and Vegetables": 500.00,
        "Medicines": 1000.00,
        "Refrigerator": 25000.00,
        "Utensils": 2000.00,
        "Lamp": 1500.00,
        "Furniture": 20000.00,
        "Streaming Subscription": 500.00,
        "Online Course": 5000.00,
        "Cloud Storage": 1000.00,
        "Sports Equipment": 3000.00,
        "Toys": 1500.00,
        "In-Game Purchases": 1000.00,
        "Pet Supplies": 2000.00,
    }

    product_categories = {
        "Retail": ["Shoes", "Smartphone", "Jewelry", "Beauty Products", "Books"],
        "Wallet": ["PhonePe Wallet", "Paytm Wallet", "Google Pay Wallet", "Amazon Pay Wallet"],
        "Consumable": ["Cleaning Products", "Personal Care Products", "Health Supplements", "Fruits and Vegetables", "Medicines"],
        "Household": ["Refrigerator", "Utensils", "Lamp", "Furniture"],
        "Services": ["Streaming Subscription", "Online Course", "Cloud Storage"],
        "Miscellaneous": ["Sports Equipment", "Toys", "In-Game Purchases", "Pet Supplies"]
    }

    product_to_category = {product: category for category, products in product_categories.items() for product in products}

    merchant_options = {
        "Wallet": ["Flipkart", "Amazon", "Google Play", "BigBasket", "Uber", "Zomato", "Swiggy Instamart"],
        "Consumable": ["BigBasket", "Blinkit", "DMart", "JioMart", "Swiggy Instamart", "Zepto", "Netmeds", "Practo", "PharmEasy", "MilkBasket"],
        "Retail": ["Flipkart", "Amazon", "Reliance Digital", "Croma", "Tata Cliq", "Myntra", "Nykaa", "Ajio", "Meesho", "Snapdeal"],
        "Household": ["IKEA", "Urban Ladder", "PepperFry", "Wakefit", "Home Centre", "Nilkamal", "Durian", "Godrej Interio", "Hometown"],
        "Services": ["Netflix", "Amazon Prime", "Hotstar", "Spotify", "Zee5", "JioSaavn", "Unacademy", "Byju's", "ALT Balaji", "Sony LIV", "Audable", "Coursera", "Udemy", "Skillshare"],
        "Miscellaneous": ["Dream11", "RummyCircle", "PokerBaazi", "MPL", "Decathlon", "FirstCry", "Tata 1mg", "1x BET", "Betway", "Lottoland", "WinZO", "Nazara Games"]
    }

    card_number_tooltip = "Enter a 16-digit card number."
    email_tooltip = "Enter a valid email address."
    phone_number_tooltip = "Phone number should start with +91 followed by a 10-digit number."
    date_time_tooltip = "Enter the time in HH:MM:SS format (e.g., 14:30:00)"

    def get_merchant_email(category, merchant):
        if merchant:
            return f"{category.lower()}@{merchant.lower()}.com".replace(' ', '')
        return ""

    col1, col2 = st.columns([6, 1])
    with col1:
        st.title("Transaction Entry Form")
    with col2:
        if st.button("👤", key="profile_button"):
            st.session_state.sidebar_open = not st.session_state.sidebar_open
            time.sleep(0.2)

    if st.session_state.sidebar_open:
        with st.sidebar:
            st.sidebar.markdown("<h2 style='color:#6c63ff;'>Profile Information 👤</h2>", unsafe_allow_html=True)

            with st.expander("⌛ **Transaction Date & Time**", expanded=True):
                st.session_state.transaction_date = st.date_input(
                    "Transaction Date",
                    value=datetime.now(),
                    min_value=datetime(2020, 1, 1),
                    max_value=datetime.now().date(),
                    help="Select the date of the transaction using the calendar."
                )

                st.session_state.transaction_time = st.text_input(
                    "Transaction Time (HH:MM:SS)",
                    value=st.session_state.transaction_time,
                    max_chars=8,
                    help=date_time_tooltip
                )

                time_pattern = re.compile(r"^\d{2}:\d{2}:\d{2}$")
                if time_pattern.match(st.session_state.transaction_time):
                    try:
                        st.session_state.transaction_dt = datetime.strptime(
                            f"{st.session_state.transaction_date} {st.session_state.transaction_time}",
                            "%Y-%m-%d %H:%M:%S"
                        ).strftime("%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        st.error("Invalid time format! Please enter a valid time (e.g., 14:30:00).")
                else:
                    st.error("Time must be in HH:MM:SS format (e.g., 14:30:00).")

            with st.expander("💳 **Card Details**", expanded=True):
                if "masked_input" not in st.session_state:
                    st.session_state.masked_input = ""
                
                st.session_state.masked_input = mask_card_number(st.session_state.card_number)
                card_number_input = st.text_input("Enter Card Number", placeholder="Enter Card Number", value=st.session_state.masked_input , help=card_number_tooltip)
                if card_number_input != st.session_state.masked_input:
                    st.session_state.card_number = card_number_input
                
                card_number = st.session_state.card_number
                bin_number = card_number[:6] if len(card_number) >= 6 else ""
                st.session_state.bin_number = bin_number

            with st.expander("📋 **Card Specifications**", expanded=True):
                card_network = st.radio("Card Network", ["Visa", "Mastercard", "American Express", "Rupay"], horizontal=True, index=["Visa", "Mastercard", "American Express", "Rupay"].index(st.session_state.card_network))
                st.session_state.card_network = card_network
                card_tier = st.radio("Card Tier", ["Silver", "Gold", "Black", "Platinum"], horizontal=True, index=["Silver", "Gold", "Black", "Platinum"].index(st.session_state.card_tier))
                st.session_state.card_tier = card_tier
                card_type = st.radio("Card Type", ["Debit", "Credit", "Prepaid"], horizontal=True, index=["Debit", "Credit", "Prepaid"].index(st.session_state.card_type))
                st.session_state.card_type = card_type

            with st.expander("📞 **Contact & Region**", expanded=True):
                user_id = st.text_input("User ID", placeholder="Enter a valid User ID", value="2200")
                if user_id:
                    if not user_id.isdigit() or len(user_id) > 6 or len(user_id) < 1:
                        if not user_id.isdigit():
                            st.error("User ID must contain only digits")
                        if len(user_id) > 6:
                            st.error("User ID must be at most 6 digits")
                        if len(user_id) < 1:
                            st.error("User ID must be at least 1 digit")
                st.session_state.user_id = user_id

                phone_number = st.text_input("Phone Number", placeholder="+91 XXXXXXXXXX", help=phone_number_tooltip, value="+91 9879879871")
                st.session_state.phone_number = phone_number
                if phone_number and not re.match(r'^\+91\s\d{10}$', phone_number):
                    st.error("Phone number must be in the format: +91 XXXXXXXXXX")

                region = [
                    "Bengaluru Urban", "Bagalkot", "Ballari", "Belagavi", "Bengaluru Rural", "Bidar",
                    "Chamarajanagar", "Chikkaballapur", "Chikkamagaluru", "Chitradurga", "Dakshina Kannada",
                    "Davanagere", "Dharwad", "Gadag", "Hassan", "Haveri", "Kalaburagi", "Kodagu", "Kolar",
                    "Koppal", "Mandya", "Mysuru", "Raichur", "Ramanagara", "Shivamogga", "Tumakuru",
                    "Udupi", "Uttara Kannada", "Vijayapura", "Yadgir"
                ]
                user_region = st.selectbox("User Region", region, index=region.index(st.session_state.user_region))
                st.session_state.user_region = user_region
                sender_email = st.text_input(
                    "Sender Email",
                    value=st.session_state.sender_email,
                    placeholder="Enter sender's email",
                    help=email_tooltip,
                    key="sender_email_input"
                )
                if sender_email != st.session_state.sender_email:
                    st.session_state.sender_email = sender_email
                if sender_email and not validate_email(sender_email):
                    st.error("Please enter a valid email address")

            with st.expander("📲 **Device Information**", expanded=True):
                device_info_to_type = {
                    "Windows": "Desktop",
                    "Linux": "Desktop",
                    "MacOS": "Desktop",
                    "iOS Device": "Mobile",
                    "Android": "Mobile",
                    "Samsung": "Mobile",
                    "Redmi": "Mobile",
                    "Realme": "Mobile",
                    "Oppo": "Mobile",
                    "Vivo": "Mobile",
                    "Motorola": "Mobile",
                    "Pixel": "Mobile",
                    "Poco": "Mobile",
                    "Huawei": "Mobile"
                }
                device_info = st.selectbox("Device Info", list(device_info_to_type.keys()))
                st.session_state.device_info = device_info
                device_type = device_info_to_type.get(device_info, "Unknown")
                st.session_state.device_type = device_type

                merchant_email = st.text_input(
                    "Merchant Email",
                    value=st.session_state.merchant_email,
                    help=email_tooltip,
                    key="merchant_email_input"
                )
                if merchant_email != st.session_state.merchant_email:
                    st.session_state.merchant_email = merchant_email

    with st.expander("🛒 **Product Category & Merchant Details**", expanded=True):
        st.subheader("Select your product")
        all_products = [product for category_products in product_categories.values() for product in category_products]
        selected_product = st.selectbox(
            "Products",
            all_products,
            index=all_products.index(st.session_state.selected_product)
        )

        if selected_product != st.session_state.selected_product:
            st.session_state.selected_product = selected_product
            st.session_state.selected_category = product_to_category[selected_product]
            st.session_state.transaction_amount = default_amounts[st.session_state.selected_product]
            merchants = merchant_options.get(st.session_state.selected_category, [])
            st.session_state.selected_merchant = merchants[0] if merchants else None
            st.session_state.merchant_email = get_merchant_email(st.session_state.selected_category, st.session_state.selected_merchant)
            st.rerun()

        category = st.session_state.selected_category
        merchants = merchant_options.get(category, [])
        selected_merchant = st.selectbox(
            "Merchant",
            merchants,
            index=merchants.index(st.session_state.selected_merchant) if st.session_state.selected_merchant in merchants else 0
        )
        if selected_merchant != st.session_state.selected_merchant:
            st.session_state.selected_merchant = selected_merchant
            st.session_state.merchant_email = get_merchant_email(category, selected_merchant)
            st.rerun()

    with st.expander("💵 **Transaction Details**", expanded=True):
        col1 = st.columns(1)[0]
        with col1:
            transaction_amt = st.number_input(
                "Transaction Amount (₹)",
                min_value=0.01,
                max_value=5000000.0,
                value=st.session_state.transaction_amount,
                format="%.2f"
            )

    with st.expander("🌎 **Order & Receiver Details**", expanded=True):
        regions = [
            "Bengaluru Urban", "Bagalkot", "Ballari", "Belagavi", "Bengaluru Rural", "Bidar",
            "Chamarajanagar", "Chikkaballapur", "Chikkamagaluru", "Chitradurga", "Dakshina Kannada",
            "Davanagere", "Dharwad", "Gadag", "Hassan", "Haveri", "Kalaburagi", "Kodagu", "Kolar",
            "Koppal", "Mandya", "Mysuru", "Raichur", "Ramanagara", "Shivamogga", "Tumakuru",
            "Udupi", "Uttara Kannada", "Vijayapura", "Yadgir"
        ]
        col1, col2 = st.columns(2)
        with col1:
            order_region = st.selectbox("Where are you ordering from?", regions)
        with col2:
            receiver_region = st.selectbox("Deliver To ", regions)

    st.divider()

    if st.button("**Submit Transaction**", use_container_width=True, key="submit_transaction_2"):
        errors = []
        if not st.session_state.transaction_dt.strip():
            errors.append("Transaction Date & Time is required!")
        if transaction_amt <= 0:
            errors.append("Transaction Amount must be greater than 0!")
        if not st.session_state.selected_category:
            errors.append("Product Category is required!")
        if not selected_merchant:
            errors.append("Merchant is required!")
        if not st.session_state.user_region:
            errors.append("User Region is required!")
        if not receiver_region:
            errors.append("Receiver Region is required!")
        if not order_region:
            errors.append("Order Region is required!")
        if not st.session_state.merchant_email.strip():
            errors.append("Merchant Email is required!")
        if not st.session_state.card_number:
            errors.append("Card Number is required!")
        if not st.session_state.bin_number:
            errors.append("BIN Number is required!")
        if not st.session_state.phone_number:
            errors.append("Phone number is required!")
        if not st.session_state.sender_email:
            errors.append("Sender Email is required!")
        if not st.session_state.user_id:
            errors.append("User ID is required!")
        if not validate_card_number(st.session_state.card_number):
            errors.append("Invalid Card Number!")
        if not validate_bin_number(st.session_state.bin_number):
            errors.append("Invalid BIN Number!")
        if not validate_email(st.session_state.sender_email):
            errors.append("Invalid Sender Email format!")
        if not validate_email(st.session_state.merchant_email):
            errors.append("Invalid Merchant Email format!")
        
        if errors:
            for error in errors:
                st.error(error)
        else:
            transaction_id = int(time.time() * 1000)
            device_info_to_type = {
                "Windows": "Desktop",
                "Linux": "Desktop",
                "MacOS": "Desktop",
                "iOS Device": "Mobile",
                "Android": "Mobile",
                "Samsung": "Mobile",
                "Redmi": "Mobile",
                "Realme": "Mobile",
                "Oppo": "Mobile",
                "Vivo": "Mobile",
                "Motorola": "Mobile",
                "Pixel": "Mobile",
                "Poco": "Mobile",
                "Huawei": "Mobile"
            }

            transaction_data = {
                "TransactionID": transaction_id,
                "TransactionAmt": float(transaction_amt),
                "TransactionDT": st.session_state.transaction_dt,
                "ProductCD": st.session_state.selected_category,
                "User_ID": int(st.session_state.user_id),
                "Merchant": selected_merchant,
                "BINNumber": st.session_state.bin_number,
                "CardNumber": st.session_state.card_number,
                "CardNetwork": st.session_state.card_network,
                "CardTier": st.session_state.card_tier,
                "CardType": st.session_state.card_type,
                "PhoneNumbers": st.session_state.phone_number,
                "User_Region": st.session_state.user_region,
                "Order_Region": order_region,
                "Receiver_Region": receiver_region,
                "Sender_email": st.session_state.sender_email,
                "Merchant_email": st.session_state.merchant_email,
                "DeviceInfo": st.session_state.device_info,
                "DeviceType": device_info_to_type.get(st.session_state.device_info, "Unknown")
            }

            success, result = send_to_backend(transaction_data)
            fraud_data = fraud_confidence_display(result)

            if success:
                st.session_state.transaction_result = result
                fraud_probability = result["fraud_detection"]["fraud_probability"]

                if fraud_probability <= 0.25:
                    st.session_state.otp_verified = True
                    st.session_state.transaction_verified = True
                    st.markdown(
                        f"""
                        <div style="background-color:#DFF2BF; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                            <h3 style="color:#4F8A10; margin: 0;"><b>Transaction Approved</b></h3>
                            <p style="color: #333; margin: 10px 0;">Your transaction has been successfully processed. No further action is required.</p>
                            <div style="
                                display: block; 
                                background-color: {fraud_data['color']}; 
                                color: white;
                                height : 35px;
                                width : 225px; 
                                padding: 5px 10px; 
                                border-radius: 5px; 
                                font-size: 14px; 
                                font-weight: bold; 
                                margin-top: 10px;">
                                {fraud_data['icon']} Identity: {fraud_data['label']} ({fraud_data['confidence']:.1f}%)
                            </div>
                            <h4 style="color: #4F8A10; margin: 10px 0 0 0;">Thank you for choosing our service.</h4>
                        </div>
                        """,
                    unsafe_allow_html=True
                    )
                    st.markdown("<div style='margin: 20px 0; text-align: center;'>", unsafe_allow_html=True)
                    fraud_meter(st.session_state.transaction_result)
                    st.markdown("</div>", unsafe_allow_html=True)
                    display_top_features(st.session_state.transaction_result)

                elif fraud_probability > 0.25 and fraud_probability <= 0.50:
                    st.markdown(
                        f"""
                        <div style="background-color:#DFF2BF; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                            <h3 style="color:#4F8A10; margin: 0;"><b>Transaction Approved</b></h3>
                            <p style="color: #333; margin: 10px 0;">Your transaction has been successfully processed. No further action is required.</p>
                            <div style="
                                display: block; 
                                background-color: {fraud_data['color']}; 
                                color: white; 
                                padding: 5px 10px;
                                height : 35px;
                                width : 250px; 
                                border-radius: 5px; 
                                font-size: 14px; 
                                font-weight: bold; 
                                margin-top: 10px;">
                                {fraud_data['icon']} Identity: {fraud_data['label']} ({fraud_data['confidence']:.1f}%)
                            </div>
                            <h4 style="color: #4F8A10; margin: 10px 0 0 0;">Thank you for choosing our service.</h4>
                        </div>
                        """,
                    unsafe_allow_html=True
                    )
                    st.markdown("<div style='margin: 20px 0; text-align: center;'>", unsafe_allow_html=True)
                    fraud_meter(st.session_state.transaction_result)
                    st.markdown("</div>", unsafe_allow_html=True)
                    display_top_features(st.session_state.transaction_result)
                    st.session_state.show_otp_page = True
                    st.rerun()

                else:
                    st.markdown(
                        f"""
                        <div style="background-color:#DFF2BF; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                            <h3 style="color:#4F8A10; margin: 0;"><b>Fraudlent Transaction Detected</b></h3>
                            <p style="color: #333; margin: 10px 0;">This transaction has been flagged as potentially fraudulent and is currently on hold.
                            Please contact our customer support team for further verification and assistance.</p>
                            <div style="
                                display: block; 
                                background-color: {fraud_data['color']}; 
                                color: white; 
                                padding: 5px 10px;
                                height : 35px;
                                width : 250px;
                                border-radius: 5px; 
                                font-size: 14px; 
                                font-weight: bold; 
                                margin-top: 10px;">
                                {fraud_data['icon']} Identity: {fraud_data['label']} ({fraud_data['confidence']:.1f}%)
                            </div>
                            <h4 style="color: #4F8A10; margin: 10px 0 0 0;">Thank you for choosing our service.</h4>
                        </div>
                        """,
                    unsafe_allow_html=True
                    )
                    st.markdown("<div style='margin: 20px 0; text-align: center;'>", unsafe_allow_html=True)
                    fraud_meter(st.session_state.transaction_result)
                    st.markdown("</div>", unsafe_allow_html=True)
                    display_top_features(st.session_state.transaction_result)
            else:
                st.error(f"Failed to submit transaction: {result}")
                st.info("Please make sure your backend API is running at http://127.0.0.1:8000")


def otp_page():
    st.title("OTP Verification Page")
    st.markdown(
                """
                <div style="background-color:#FFF4CC; padding: 15px; border-radius: 10px;">
                    <h3 style="color:#9F6000;"><b>Suspicious Transaction Detected</b></h3>
                    <p>This transaction requires additional verification. An OTP has been sent to your registered mobile number.</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
    fraud_meter(st.session_state.transaction_result)
    display_top_features(st.session_state.transaction_result) 
    
    user_otp = st.text_input("Enter 6-digit OTP", max_chars=6, type="password", key="user_otp")
     
    
    if st.button("Verify OTP"):
        if user_otp == "123456":  # Replace with actual OTP logic
            st.session_state.otp_verified = True
            st.session_state.show_otp_page = False  # Return to main transaction page
            st.markdown(
                        """
                        <div style="background-color:#DFF2BF; padding: 15px; border-radius: 10px;">
                            <h3 style="color:#4F8A10;"><b>OTP Verified Successfully</b></h3>
                            <p>Your OTP has been successfully verified. The transaction has been approved.</p>
                            <h4>Thank you for choosing our service.</h4>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
            
            # Show Complete Transaction button after OTP verification
            if st.button("Complete Transaction"):
                st.session_state.show_otp_page = False  # Navigate back to the main page
                st.session_state.transaction_verified = True
                st.success("Transaction completed!")
        else:
            st.session_state.otp_verified = False
            st.error("Invalid OTP! Please try again.")

def main():
    region = [
        "Bengaluru Urban", "Bagalkot", "Ballari", "Belagavi", "Bengaluru Rural", "Bidar",
        "Chamarajanagar", "Chikkaballapur", "Chikkamagaluru", "Chitradurga", "Dakshina Kannada",
        "Davanagere", "Dharwad", "Gadag", "Hassan", "Haveri", "Kalaburagi", "Kodagu", "Kolar",
        "Koppal", "Mandya", "Mysuru", "Raichur", "Ramanagara", "Shivamogga", "Tumakuru",
        "Udupi", "Uttara Kannada", "Vijayapura", "Yadgir"
    ]
    device_info_to_type = {
        "Windows": "Desktop",
        "Linux": "Desktop",
        "MacOS": "Desktop",
        "iOS Device": "Mobile",
        "Android": "Mobile",
        "Samsung": "Mobile",
        "Redmi": "Mobile",
        "Realme": "Mobile",
        "Oppo": "Mobile",
        "Vivo": "Mobile",
        "Motorola": "Mobile",
        "Pixel": "Mobile",
        "Poco": "Mobile",
        "Huawei": "Mobile"
    }
    if "otp" not in st.session_state:
        st.session_state.otp = "123456"
    if "otp_verified" not in st.session_state:
        st.session_state.otp_verified = False
    if "transaction_result" not in st.session_state:
        st.session_state.transaction_result = None
    if "user_otp" not in st.session_state:
        st.session_state.user_otp = ""
    if "transaction_verified" not in st.session_state:
        st.session_state.transaction_verified = False
    if 'show_otp_page' not in st.session_state:
        st.session_state.show_otp_page = False
    if "transaction_time" not in st.session_state:
        st.session_state.transaction_time = datetime.now().strftime("%H:%M:%S")
    if "transaction_dt" not in st.session_state:
        st.session_state.transaction_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if "transaction_date" not in st.session_state:
        st.session_state.transaction_date = datetime.now().date()
    if "card_number" not in st.session_state:
        st.session_state.card_number = "1234567890123456"
    if "card_network" not in st.session_state:
        st.session_state.card_network = "Rupay"
    if "card_tier" not in st.session_state:
        st.session_state.card_tier = "Black"
    if "card_type" not in st.session_state:
        st.session_state.card_type = "Credit"
    if "phone_number" not in st.session_state:
        st.session_state.phone_number = "+91 9879879871"
    if "user_id" not in st.session_state:
        st.session_state.user_id = "2200"
    if "user_region" not in st.session_state:
        st.session_state.user_region = region[0]
    if "device_info" not in st.session_state:
        st.session_state.device_info = "Redmi"
    if "device_type" not in st.session_state:
        st.session_state.device_type = device_info_to_type.get(st.session_state.device_info, "Unknown")

    if st.session_state.show_otp_page:
        otp_page()
    else:
        transaction_page()

if __name__ == "__main__":
    main()