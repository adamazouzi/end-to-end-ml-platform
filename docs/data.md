# Dataset documentation

## Source and terms

This project uses IBM's `Telco-Customer-Churn.csv` from the archived [IBM telco-customer-churn-on-icp4d repository](https://github.com/IBM/telco-customer-churn-on-icp4d), downloaded from its [raw CSV](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv). IBM's [README](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/README.md) identifies it as a Telco customer churn dataset and uses it in a churn modeling code pattern. The README says that code pattern is licensed under Apache-2.0 and that separate third-party components follow their own licenses; it does not clearly grant a separate license to the CSV. The repository's [Apache-2.0 LICENSE](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/LICENSE) therefore must not be assumed to settle the CSV's rights. This project records IBM provenance and the fetched file's checksum but does not redistribute the dataset. Review upstream terms before redistribution or commercial use.

The download workflow records the SHA-256 of the exact fetched file in its command output. We do not pin a checksum from a third-party mirror as an upstream integrity guarantee. Dataset content is kept under `data/raw/` and excluded from Git.

## Expected data

The source CSV has 7,043 customer rows and 21 columns (one identifier, 19 candidate features, and the target), subject to upstream changes. Validation checks the exact required field set, identifier uniqueness, nonempty target, numeric domains, and duplicate rows. Eleven rows in the commonly used source have a blank `TotalCharges` value for new customers; the feature pipeline parses these as missing and imputes from training data.

| Field | Meaning / expected values |
| --- | --- |
| `customerID` | Unique customer identifier; excluded from model features. |
| `gender` | Female or Male. |
| `SeniorCitizen` | 0 or 1. |
| `Partner`, `Dependents` | Yes or No. |
| `tenure` | Months with company, 0–72. |
| `PhoneService` | Yes or No. |
| `MultipleLines` | Yes, No, or No phone service. |
| `InternetService` | DSL, Fiber optic, or No. |
| `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` | Yes, No, or No internet service. |
| `Contract` | Month-to-month, One year, or Two year. |
| `PaperlessBilling` | Yes or No. |
| `PaymentMethod` | Electronic check, Mailed check, Bank transfer (automatic), or Credit card (automatic). |
| `MonthlyCharges`, `TotalCharges` | Non-negative numeric amounts; blank total charges are imputed. |
| `Churn` | Target label: Yes or No. |

Fields listed here document the expected source schema. Validation currently enforces required column presence and numeric/target domains; categorical vocabularies are preserved for model encoding and API input validation.

## Known limitations

The dataset is a sample for one Telco customer-churn use case. The provenance information documented here does not establish its collection period or representativeness for current customers or other markets. Performance results should not be treated as production forecasts. Customer identifiers are never used for training.
