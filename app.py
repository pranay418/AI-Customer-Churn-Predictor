import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os

# Set page configuration
st.set_page_config(
    page_title="AI Customer Churn Predictor",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern premium dashboard aesthetics
st.markdown("""
<style>
    /* Gradient Background and general page styling */
    .stApp {
        background-color: #0B0F19 !important;
        color: #F3F4F6 !important;
    }
    
    /* Metrics panel custom container */
    div[data-testid="metric-container"] {
        background-color: #162032 !important;
        border: 1px solid #233554 !important;
        border-radius: 12px;
        padding: 18px 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: transform 0.2s ease-in-out;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        border-color: #3b82f6 !important;
    }
    
    /* Force high-contrast text for widget labels (Gender, SeniorCitizen, etc.) */
    [data-testid="stWidgetLabel"] p, .stApp label, .stApp p, .stApp span {
        color: #E5E7EB !important;
        font-weight: 500 !important;
    }
    
    /* Ensure input values and dropdown values are highly readable */
    div[data-baseweb="select"] div, input, select, textarea {
        color: #FFFFFF !important;
        font-weight: 500 !important;
    }
    
    /* Sidebar specific text readability */
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        color: #F3F4F6 !important;
        font-weight: 600 !important;
    }
    section[data-testid="stSidebar"] h4 {
        color: #FFFFFF !important;
    }
    
    /* Headers and labels styling */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        background: linear-gradient(135deg, #60A5FA 0%, #3B82F6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Override plain HTML headers styling for high contrast */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
        color: #FFFFFF !important;
    }
    
    /* Tabs selection active/inactive styling */
    button[data-baseweb="tab"] p {
        color: #9CA3AF !important;
        font-size: 1.05rem !important;
    }
    button[aria-selected="true"][data-baseweb="tab"] p {
        color: #3B82F6 !important;
        font-weight: 700 !important;
    }
    
    /* Cards and boxes styling */
    .dashboard-card {
        background-color: #111A2E !important;
        border: 1px solid #1E2D4A !important;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
    }
    
    /* Sidebar custom style */
    section[data-testid="stSidebar"] {
        background-color: #0D1527 !important;
        border-right: 1px solid #1E2D4A !important;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions to load model and data
@st.cache_resource
def load_model():
    if not os.path.exists("model.joblib"):
        return None
    return joblib.load("model.joblib")

@st.cache_data
def load_data():
    if not os.path.exists("WA_Fn-UseC_-Telco-Customer-Churn.csv"):
        return None
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(' ', np.nan), errors='coerce').fillna(0)
    return df

# Load assets
model_data = load_model()
df = load_data()

# Check if model and dataset are present
if model_data is None or df is None:
    st.error("Missing project assets! Please ensure both 'WA_Fn-UseC_-Telco-Customer-Churn.csv' and 'model.joblib' are present in the directory.")
    st.info("Run `python download_data.py` first, followed by `python train.py` to generate the required model and data assets.")
    st.stop()

pipeline = model_data['pipeline']
feature_importance = model_data['feature_importance']
categorical_options = model_data['categorical_options']

# App Title
st.title("🔮 AI Customer Churn Predictor")
st.markdown("##### SaaS & Subscription Customer Retention Intelligence Platform")

# Top KPI Panel
churn_rate = (df['Churn'] == 'Yes').mean() * 100
total_customers = len(df)
avg_monthly_charges = df['MonthlyCharges'].mean()
avg_tenure = df['tenure'].mean()

kpi_cols = st.columns(4)
with kpi_cols[0]:
    st.metric(label="Total Customers Analyzed", value=f"{total_customers:,}", delta=None)
with kpi_cols[1]:
    st.metric(label="Average Churn Rate", value=f"{churn_rate:.2f}%", delta="-1.2% MoM", delta_color="inverse")
with kpi_cols[2]:
    st.metric(label="Avg Monthly Charges", value=f"${avg_monthly_charges:.2f}", delta=None)
with kpi_cols[3]:
    st.metric(label="Average Tenure", value=f"{avg_tenure:.1f} Months", delta=None)

# Create Sidebar for individual customer inputs
st.sidebar.header("👤 Customer Profile Builder")
st.sidebar.write("Configure customer attributes to predict real-time churn risk.")

# User inputs in sidebar
input_data = {}

# Demographics Section
st.sidebar.markdown("---")
st.sidebar.markdown("#### **Demographics**")
input_data['gender'] = st.sidebar.selectbox("Gender", categorical_options['gender'])
input_data['SeniorCitizen'] = st.sidebar.selectbox("Senior Citizen?", categorical_options['SeniorCitizen'])
input_data['Partner'] = st.sidebar.selectbox("Has Partner?", categorical_options['Partner'])
input_data['Dependents'] = st.sidebar.selectbox("Has Dependents?", categorical_options['Dependents'])

# Subscriptions Section
st.sidebar.markdown("---")
st.sidebar.markdown("#### **Subscription Details**")
input_data['tenure'] = st.sidebar.slider("Tenure (Months)", min_value=0, max_value=72, value=12)
input_data['Contract'] = st.sidebar.selectbox("Contract Type", categorical_options['Contract'])
input_data['PaperlessBilling'] = st.sidebar.selectbox("Paperless Billing?", categorical_options['PaperlessBilling'])
input_data['PaymentMethod'] = st.sidebar.selectbox("Payment Method", categorical_options['PaymentMethod'])

# Services Section
st.sidebar.markdown("---")
st.sidebar.markdown("#### **Services Subscribed**")
input_data['PhoneService'] = st.sidebar.selectbox("Phone Service?", categorical_options['PhoneService'])

# If no Phone Service, MultipleLines is forced to 'No phone service'
if input_data['PhoneService'] == 'No':
    input_data['MultipleLines'] = 'No phone service'
else:
    input_data['MultipleLines'] = st.sidebar.selectbox("Multiple Lines?", [opt for opt in categorical_options['MultipleLines'] if opt != 'No phone service'])

input_data['InternetService'] = st.sidebar.selectbox("Internet Service Provider", categorical_options['InternetService'])

if input_data['InternetService'] == 'No':
    input_data['OnlineSecurity'] = 'No internet service'
    input_data['OnlineBackup'] = 'No internet service'
    input_data['DeviceProtection'] = 'No internet service'
    input_data['TechSupport'] = 'No internet service'
    input_data['StreamingTV'] = 'No internet service'
    input_data['StreamingMovies'] = 'No internet service'
else:
    input_data['OnlineSecurity'] = st.sidebar.selectbox("Online Security", [opt for opt in categorical_options['OnlineSecurity'] if opt != 'No internet service'])
    input_data['OnlineBackup'] = st.sidebar.selectbox("Online Backup", [opt for opt in categorical_options['OnlineBackup'] if opt != 'No internet service'])
    input_data['DeviceProtection'] = st.sidebar.selectbox("Device Protection", [opt for opt in categorical_options['DeviceProtection'] if opt != 'No internet service'])
    input_data['TechSupport'] = st.sidebar.selectbox("Tech Support", [opt for opt in categorical_options['TechSupport'] if opt != 'No internet service'])
    input_data['StreamingTV'] = st.sidebar.selectbox("Streaming TV", [opt for opt in categorical_options['StreamingTV'] if opt != 'No internet service'])
    input_data['StreamingMovies'] = st.sidebar.selectbox("Streaming Movies", [opt for opt in categorical_options['StreamingMovies'] if opt != 'No internet service'])

# Financials Section
st.sidebar.markdown("---")
st.sidebar.markdown("#### **Financials**")
# Restrict Monthly Charges slider according to selected internet option for realism
min_c, max_c, val_c = 18.0, 120.0, 70.0
if input_data['InternetService'] == 'No':
    max_c = 30.0
    val_c = 20.0
elif input_data['InternetService'] == 'DSL':
    max_c = 90.0
    val_c = 55.0
else: # Fiber optic
    min_c = 65.0
    val_c = 90.0

input_data['MonthlyCharges'] = st.sidebar.slider("Monthly Charges ($)", min_value=float(min_c), max_value=float(max_c), value=float(val_c))

# Automatically calculate TotalCharges as tenure * MonthlyCharges, but allow custom overrides
total_calculated = input_data['MonthlyCharges'] * input_data['tenure']
input_data['TotalCharges'] = st.sidebar.number_input("Total Charges ($)", min_value=0.0, value=float(total_calculated), step=50.0)

# Create Main tabs
tab1, tab2 = st.tabs(["🎯 Customer Churn Risk Predictor", "📊 Global Churn Analytics"])

# ----------------- TAB 1: PREDICTIONS -----------------
with tab1:
    st.markdown("### Active Customer Risk Assessment")
    
    # Predict probability
    input_df = pd.DataFrame([input_data])
    prediction_prob = pipeline.predict_proba(input_df)[0][1]
    prediction_class = pipeline.predict(input_df)[0]
    
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
        st.subheader("Churn Probability Meter")
        
        # Plot Gauge Chart
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prediction_prob * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Probability of Unsubscribing", 'font': {'size': 18, 'color': '#F3F4F6'}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#F3F4F6"},
                'bar': {'color': "#EF4444" if prediction_prob > 0.5 else "#F59E0B" if prediction_prob > 0.25 else "#10B981"},
                'bgcolor': "#162032",
                'borderwidth': 2,
                'bordercolor': "#233554",
                'steps': [
                    {'range': [0, 25], 'color': '#064e3b'},
                    {'range': [25, 50], 'color': '#78350f'},
                    {'range': [50, 100], 'color': '#7f1d1d'}
                ],
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': "#F3F4F6", 'family': "Inter"},
            height=280,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Risk Badge and Recommendation
        if prediction_prob > 0.5:
            st.error("🚨 **Classification: HIGH RISK OF CHURN**")
            risk_color = "red"
        elif prediction_prob > 0.25:
            st.warning("⚠️ **Classification: MEDIUM RISK OF CHURN**")
            risk_color = "orange"
        else:
            st.success("✅ **Classification: LOW RISK OF CHURN (Healthy)**")
            risk_color = "green"
            
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Risk Explanations Card
        st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
        st.subheader("💡 Key Risk Factors & Recommendations")
        
        recs = []
        if input_data['Contract'] == 'Month-to-month':
            recs.append("🔴 **Month-to-month Contract**: This is the strongest driver of churn. *Recommendation: Offer a 10% discount on a 1-year contract conversion.*")
        if input_data['InternetService'] == 'Fiber optic':
            recs.append("🟡 **Fiber Optic Service**: Fiber optic users show higher churn due to price sensitivity. *Recommendation: Offer a bundled value addon (e.g. streaming or cloud backup) or check for service issues.*")
        if input_data['tenure'] < 6:
            recs.append("🔴 **New Customer Onboarding Phase**: Customer is in high-risk early tenure (<6 months). *Recommendation: Schedule a proactive check-in call or onboarding support.*")
        if input_data['TechSupport'] == 'No' and input_data['InternetService'] != 'No':
            recs.append("🟡 **No Tech Support**: Lack of tech support is strongly associated with churn. *Recommendation: Promote free trial or discount on Tech Support addon.*")
        if input_data['OnlineSecurity'] == 'No' and input_data['InternetService'] != 'No':
            recs.append("🟡 **No Online Security**: Security features bind customers longer. *Recommendation: Offer a basic Online Security package.*")
        
        if len(recs) == 0:
            st.write("This customer has a stable profile (long tenure, long-term contract, multiple security features). Keep up standard maintenance!")
        else:
            for rec in recs:
                st.markdown(rec)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
        st.subheader("📊 Customer Feature Comparison")
        st.write("Comparing this customer's parameters against the average active and churned customers in the dataset.")
        
        # Sub-dataframes for comparisons
        df_churned = df[df['Churn'] == 'Yes']
        df_retained = df[df['Churn'] == 'No']
        
        comp_df = pd.DataFrame({
            'Metric': ['Tenure (Months)', 'Monthly Charges ($)', 'Total Charges ($)'],
            'Selected Customer': [input_data['tenure'], input_data['MonthlyCharges'], input_data['TotalCharges']],
            'Avg Churned Customer': [df_churned['tenure'].mean(), df_churned['MonthlyCharges'].mean(), df_churned['TotalCharges'].mean()],
            'Avg Active Customer': [df_retained['tenure'].mean(), df_retained['MonthlyCharges'].mean(), df_retained['TotalCharges'].mean()]
        })
        
        # Melt the dataframe for plotly plotting
        comp_melt = pd.melt(comp_df, id_vars=['Metric'], value_vars=['Selected Customer', 'Avg Churned Customer', 'Avg Active Customer'],
                            var_name='Group', value_name='Value')
        
        # Render a bar chart of comparison metrics
        fig_comp = px.bar(
            comp_melt,
            x='Metric',
            y='Value',
            color='Group',
            barmode='group',
            color_discrete_map={
                'Selected Customer': '#3B82F6',
                'Avg Churned Customer': '#EF4444',
                'Avg Active Customer': '#10B981'
            },
            template="plotly_dark"
        )
        
        fig_comp.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=10, r=10, t=10, b=10),
            height=300
        )
        st.plotly_chart(fig_comp, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Quick summary breakdown
        st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
        st.subheader("📋 Customer Details Snapshot")
        snap_cols = st.columns(3)
        with snap_cols[0]:
            st.markdown(f"**Contract:**\n{input_data['Contract']}")
            st.markdown(f"**Internet:**\n{input_data['InternetService']}")
        with snap_cols[1]:
            st.markdown(f"**Payment:**\n{input_data['PaymentMethod']}")
            st.markdown(f"**Tech Support:**\n{input_data['TechSupport']}")
        with snap_cols[2]:
            st.markdown(f"**Paperless Bill:**\n{input_data['PaperlessBilling']}")
            st.markdown(f"**Security Addon:**\n{input_data['OnlineSecurity']}")
        st.markdown("</div>", unsafe_allow_html=True)

# ----------------- TAB 2: GLOBAL ANALYTICS -----------------
with tab2:
    st.markdown("### Why Do Users Churn? (Global Analysis)")
    st.write("Insights derived from historical data of 7,043 subscribers to pinpoint structural churn drivers.")
    
    col_feat1, col_feat2 = st.columns([1, 1])
    
    with col_feat1:
        st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
        st.subheader("🧠 Model-Derived Feature Importance")
        st.write("Top factors that the machine learning algorithm uses to distinguish churned vs. retained subscribers.")
        
        # Process feature importance data
        feat_df = pd.DataFrame(feature_importance)
        
        # Map feature names to user-friendly titles
        feat_df['Feature_Clean'] = feat_df['Feature'].apply(lambda x: 
            x.replace('Contract_', 'Contract: ')
             .replace('InternetService_', 'Internet: ')
             .replace('PaymentMethod_', 'Payment: ')
             .replace('OnlineSecurity_', 'Security: ')
             .replace('TechSupport_', 'Support: ')
             .replace('_', ' ')
        )
        
        fig_feat = px.bar(
            feat_df.head(10),
            x='Importance',
            y='Feature_Clean',
            orientation='h',
            color='Importance',
            color_continuous_scale='Blues',
            template='plotly_dark'
        )
        fig_feat.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(autorange="reversed"),
            coloraxis_showscale=False,
            height=300,
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig_feat, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_feat2:
        st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
        st.subheader("📅 Churn Rate by Contract Type")
        st.write("Contract model has a massive impact. Month-to-month contracts are highly volatile.")
        
        contract_churn = df.groupby('Contract')['Churn'].value_counts(normalize=True).unstack() * 100
        contract_churn = contract_churn.reset_index()
        
        fig_contract = px.bar(
            contract_churn,
            x='Contract',
            y='Yes',
            text=contract_churn['Yes'].apply(lambda x: f"{x:.1f}%"),
            color='Contract',
            color_discrete_sequence=['#EF4444', '#F59E0B', '#10B981'],
            template='plotly_dark',
            labels={'Yes': 'Churn Rate (%)'}
        )
        fig_contract.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            height=300,
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig_contract, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    col_chart3, col_chart4 = st.columns([1.2, 1])
    
    with col_chart3:
        st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
        st.subheader("💵 Monthly Charges & Tenure Density Analysis")
        st.write("A scatter view of active vs churned customers showing high concentration of churn at early tenure and higher charges.")
        
        # Sample for plotting speed if needed, but 7000 rows works fine in Plotly
        fig_scatter = px.scatter(
            df,
            x="tenure",
            y="MonthlyCharges",
            color="Churn",
            color_discrete_map={"Yes": "#EF4444", "No": "#10B981"},
            opacity=0.35,
            labels={"tenure": "Tenure (Months)", "MonthlyCharges": "Monthly Charges ($)"},
            template='plotly_dark'
        )
        fig_scatter.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=320,
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_chart4:
        st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
        st.subheader("🌐 Churn by Internet Service Provider")
        st.write("Fiber Optic customers churn at a significantly higher rate than DSL or No-Internet users.")
        
        internet_churn = df.groupby('InternetService')['Churn'].value_counts(normalize=True).unstack() * 100
        internet_churn = internet_churn.reset_index()
        
        fig_internet = px.bar(
            internet_churn,
            x='InternetService',
            y='Yes',
            text=internet_churn['Yes'].apply(lambda x: f"{x:.1f}%"),
            color='InternetService',
            color_discrete_sequence=['#FF7F50', '#87CEFA', '#32CD32'],
            template='plotly_dark',
            labels={'Yes': 'Churn Rate (%)'}
        )
        fig_internet.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            height=320,
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig_internet, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# Footer info
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #6B7280; font-size: 14px;'>"
    "AI Customer Churn Predictor app | Built for SaaS & Startups recruiters demonstrating ML classification, pipeline engineering, and custom KPI dashboards."
    "</div>",
    unsafe_allow_html=True
)
