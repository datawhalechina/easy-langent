// Mock data from telco_data.csv (first 10 rows)
export interface TelcoData {
  customerID: string;
  gender: string;
  SeniorCitizen: number;
  Partner: string;
  Dependents: string;
  tenure: number;
  PhoneService: string;
  MultipleLines: string;
  InternetService: string;
  OnlineSecurity: string;
  OnlineBackup: string;
  DeviceProtection: string;
  TechSupport: string;
  StreamingTV: string;
  StreamingMovies: string;
  Contract: string;
  PaperlessBilling: string;
  PaymentMethod: string;
  MonthlyCharges: number;
  TotalCharges: number;
  Churn: string;
}

export const mockTelcoData: TelcoData[] = [
  {
    customerID: "7590-VHVEG",
    gender: "Female",
    SeniorCitizen: 0,
    Partner: "Yes",
    Dependents: "No",
    tenure: 1,
    PhoneService: "No",
    MultipleLines: "No phone service",
    InternetService: "DSL",
    OnlineSecurity: "No",
    OnlineBackup: "Yes",
    DeviceProtection: "No",
    TechSupport: "No",
    StreamingTV: "No",
    StreamingMovies: "No",
    Contract: "Month-to-month",
    PaperlessBilling: "Yes",
    PaymentMethod: "Electronic check",
    MonthlyCharges: 29.85,
    TotalCharges: 29.85,
    Churn: "No"
  },
  {
    customerID: "5575-GNVDE",
    gender: "Male",
    SeniorCitizen: 0,
    Partner: "No",
    Dependents: "No",
    tenure: 34,
    PhoneService: "Yes",
    MultipleLines: "No",
    InternetService: "DSL",
    OnlineSecurity: "Yes",
    OnlineBackup: "No",
    DeviceProtection: "Yes",
    TechSupport: "No",
    StreamingTV: "No",
    StreamingMovies: "No",
    Contract: "One year",
    PaperlessBilling: "No",
    PaymentMethod: "Mailed check",
    MonthlyCharges: 56.95,
    TotalCharges: 1889.5,
    Churn: "No"
  },
  {
    customerID: "3668-QPYBK",
    gender: "Male",
    SeniorCitizen: 0,
    Partner: "No",
    Dependents: "No",
    tenure: 2,
    PhoneService: "Yes",
    MultipleLines: "No",
    InternetService: "DSL",
    OnlineSecurity: "Yes",
    OnlineBackup: "Yes",
    DeviceProtection: "No",
    TechSupport: "No",
    StreamingTV: "No",
    StreamingMovies: "No",
    Contract: "Month-to-month",
    PaperlessBilling: "Yes",
    PaymentMethod: "Mailed check",
    MonthlyCharges: 53.85,
    TotalCharges: 108.15,
    Churn: "Yes"
  },
  {
    customerID: "7795-CFOCW",
    gender: "Male",
    SeniorCitizen: 0,
    Partner: "No",
    Dependents: "No",
    tenure: 45,
    PhoneService: "No",
    MultipleLines: "No phone service",
    InternetService: "DSL",
    OnlineSecurity: "Yes",
    OnlineBackup: "No",
    DeviceProtection: "Yes",
    TechSupport: "Yes",
    StreamingTV: "No",
    StreamingMovies: "No",
    Contract: "One year",
    PaperlessBilling: "No",
    PaymentMethod: "Bank transfer (automatic)",
    MonthlyCharges: 42.30,
    TotalCharges: 1840.75,
    Churn: "No"
  },
  {
    customerID: "9237-HQITU",
    gender: "Female",
    SeniorCitizen: 0,
    Partner: "No",
    Dependents: "No",
    tenure: 2,
    PhoneService: "Yes",
    MultipleLines: "No",
    InternetService: "Fiber optic",
    OnlineSecurity: "No",
    OnlineBackup: "No",
    DeviceProtection: "No",
    TechSupport: "No",
    StreamingTV: "No",
    StreamingMovies: "No",
    Contract: "Month-to-month",
    PaperlessBilling: "Yes",
    PaymentMethod: "Electronic check",
    MonthlyCharges: 70.70,
    TotalCharges: 151.65,
    Churn: "Yes"
  },
  {
    customerID: "9305-CDSKC",
    gender: "Female",
    SeniorCitizen: 0,
    Partner: "No",
    Dependents: "No",
    tenure: 8,
    PhoneService: "Yes",
    MultipleLines: "Yes",
    InternetService: "Fiber optic",
    OnlineSecurity: "No",
    OnlineBackup: "No",
    DeviceProtection: "Yes",
    TechSupport: "No",
    StreamingTV: "Yes",
    StreamingMovies: "Yes",
    Contract: "Month-to-month",
    PaperlessBilling: "Yes",
    PaymentMethod: "Electronic check",
    MonthlyCharges: 99.65,
    TotalCharges: 820.5,
    Churn: "Yes"
  },
  {
    customerID: "1452-KIOVK",
    gender: "Male",
    SeniorCitizen: 0,
    Partner: "No",
    Dependents: "Yes",
    tenure: 22,
    PhoneService: "Yes",
    MultipleLines: "Yes",
    InternetService: "Fiber optic",
    OnlineSecurity: "No",
    OnlineBackup: "Yes",
    DeviceProtection: "No",
    TechSupport: "No",
    StreamingTV: "Yes",
    StreamingMovies: "No",
    Contract: "Month-to-month",
    PaperlessBilling: "Yes",
    PaymentMethod: "Credit card (automatic)",
    MonthlyCharges: 89.10,
    TotalCharges: 1949.4,
    Churn: "No"
  },
  {
    customerID: "6713-OKOMC",
    gender: "Female",
    SeniorCitizen: 0,
    Partner: "No",
    Dependents: "No",
    tenure: 10,
    PhoneService: "No",
    MultipleLines: "No phone service",
    InternetService: "DSL",
    OnlineSecurity: "Yes",
    OnlineBackup: "No",
    DeviceProtection: "No",
    TechSupport: "No",
    StreamingTV: "No",
    StreamingMovies: "No",
    Contract: "Month-to-month",
    PaperlessBilling: "No",
    PaymentMethod: "Mailed check",
    MonthlyCharges: 29.75,
    TotalCharges: 301.9,
    Churn: "No"
  },
  {
    customerID: "7892-POOKP",
    gender: "Female",
    SeniorCitizen: 0,
    Partner: "Yes",
    Dependents: "No",
    tenure: 28,
    PhoneService: "Yes",
    MultipleLines: "Yes",
    InternetService: "Fiber optic",
    OnlineSecurity: "No",
    OnlineBackup: "No",
    DeviceProtection: "Yes",
    TechSupport: "Yes",
    StreamingTV: "Yes",
    StreamingMovies: "Yes",
    Contract: "Month-to-month",
    PaperlessBilling: "Yes",
    PaymentMethod: "Electronic check",
    MonthlyCharges: 104.80,
    TotalCharges: 3046.05,
    Churn: "Yes"
  },
  {
    customerID: "6388-TABGU",
    gender: "Male",
    SeniorCitizen: 0,
    Partner: "No",
    Dependents: "Yes",
    tenure: 62,
    PhoneService: "Yes",
    MultipleLines: "No",
    InternetService: "DSL",
    OnlineSecurity: "Yes",
    OnlineBackup: "Yes",
    DeviceProtection: "No",
    TechSupport: "No",
    StreamingTV: "No",
    StreamingMovies: "No",
    Contract: "One year",
    PaperlessBilling: "No",
    PaymentMethod: "Bank transfer (automatic)",
    MonthlyCharges: 56.15,
    TotalCharges: 3487.95,
    Churn: "No"
  }
];

// Column information for analysis
export const columnInfo = {
  // Categorical/Discrete variables
  categorical: ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn'],
  
  // Numerical/Continuous variables
  numerical: ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
};

// Mock correlation results and conclusions
export const mockAnalysisResults = {
  'gender_Churn': {
    correlation: 0.15,
    conclusion: '性别对流失有轻微影响'
  },
  'tenure_MonthlyCharges': {
    correlation: -0.42,
    conclusion: '资历越长，月费用略低'
  },
  'Contract_TotalCharges': {
    correlation: 0.68,
    conclusion: '合同类型与总费用有较强相关性'
  },
  'InternetService_MonthlyCharges': {
    correlation: 0.73,
    conclusion: '网络服务类型显著影响月费用'
  },
  'tenure_TotalCharges': {
    correlation: 0.82,
    conclusion: '资历与总费用呈强正相关'
  },
  'gender_MonthlyCharges': {
    correlation: 0.08,
    conclusion: '性别与月费用相关性较弱'
  }
};