# DataScienceEDA
his repository offers Python scripts for mastering Exploratory Data Analysis (EDA). It covers data manipulation, time series analysis, text processing, advanced visualizations, anomaly detection, feature importance, automated EDA, and big data EDA using pandas, Dask, Vaex, Modin, Plotly, and scikit-learn. 
EDA Learning Repository
Overview
This repository provides Python scripts for mastering Exploratory Data Analysis (EDA). It covers data manipulation, time series analysis, text processing, advanced visualizations, anomaly detection, feature importance, automated EDA, and big data EDA using pandas, Dask, Vaex, Modin, Plotly, and scikit-learn. Each script includes tutorials, code examples, and up to 15 exercises for hands-on learning. Synthetic datasets (orders.csv, products.csv, employees.csv, etc.) with generation scripts are included for practice. Ideal for learners and data scientists aiming to build skills in analyzing small to large-scale datasets efficiently.
Word Count: 96 words
Installation

Clone the repository:git clone https://github.com/your-username/eda-learning-repository.git
cd eda-learning-repository


Install required Python libraries:pip install pandas numpy scikit-learn plotly dask vaex modin[ray] ydata-profiling sweetviz dtale


Ensure the data/ directory exists:mkdir data



Usage

Generate Datasets:

Run data generation scripts to create synthetic datasets:python generate_orders_csv.py
python generate_products_csv.py
python generate_employees_csv.py


Outputs: data/orders.csv, data/products.csv, data/employees.csv.
Note: Generate sales.csv and financials.csv if needed (scripts available upon request).


Run EDA Scripts:

Navigate to the advanced/ directory and execute scripts, e.g.:python advanced/Big_Data_EDA.py


Each script includes exercises and visualizations for learning.


View Outputs:

Check console outputs, HTML reports (e.g., pandas-profiling, Sweetviz), or interactive D-Tale interfaces.
Visualizations require a Jupyter notebook or browser for Plotly.



Directory Structure
├── advanced/                    # EDA scripts
│   ├── 3.Functions.py
│   ├── Time_Series_Analysis.py
│   ├── Text_Data_Analysis.py
│   ├── Advanced_Visualizations.py
│   ├── Anomaly_Detection.py
│   ├── Feature_Importance_Analysis.py
│   ├── EDA_Automation.py
│   ├── Big_Data_EDA.py
├── data/                        # Synthetic datasets
│   ├── orders.csv
│   ├── products.csv
│   ├── employees.csv
├── generate_orders_csv.py       # Data generation scripts
├── generate_products_csv.py
├── generate_employees_csv.py
├── README.md                    # Repository documentation

Contributing
Contributions are welcome! To contribute:

Fork the repository.
Create a feature branch (git checkout -b feature/new-eda-script).
Commit changes (git commit -m "Add new EDA script").
Push to the branch (git push origin feature/new-eda-script).
Open a pull request.

License
This repository is licensed under the MIT License.
