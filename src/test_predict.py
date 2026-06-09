import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from predict import predict_single

result = predict_single({
    'gender': 'Female',
    'SeniorCitizen': 1,
    'Partner': 'Yes',
    'Dependents': 'Yes',
    'tenure': 12,
    'PhoneService': 'Yes',
    'MultipleLines': 'No',
    'InternetService': 'Fiber optic',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'No',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'No',
    'StreamingMovies': 'No',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 70.5,
    'TotalCharges': 846.0
})
print(result)