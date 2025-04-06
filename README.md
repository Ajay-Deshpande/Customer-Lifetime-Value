# Customer Lifetime Value Prediction with Probabilistic Models

This project focuses on predicting Customer Lifetime Value (CLV) for an online retail company over 6-month and 12-month periods. It utilizes probabilistic models from the `lifetimes` library in Python, specifically the Buy Till You Die (BG/NBD) model for predicting the number of future purchases and the Gamma-Gamma model for estimating the average transaction value.

## Business Problem

An online company aims to understand the potential value of each customer over the next 6 and 12 months. This information can be crucial for targeted marketing efforts, customer segmentation, and resource allocation to maximize profitability.

## Methodology

This project employs a probabilistic approach to CLV prediction:

1.  **Data Preparation:** The initial online retail transaction data is cleaned and preprocessed to handle missing values, outliers, and irrelevant transactions (e.g., returns).
2.  **RFM Feature Engineering:** Using the `lifetimes` library, Recency, Frequency, Monetary Value (RFM), and Time (T) features are calculated for each customer based on their transaction history.
3.  **BG/NBD Model:** The Beta-Geometric/Negative Binomial Distribution (BG/NBD) model is fitted to the frequency, recency, and T data to predict the expected number of future purchases for each customer.
4.  **Gamma-Gamma Model:** The Gamma-Gamma model is applied to the frequency and monetary value data to estimate the average transaction value for each customer in their future purchases. This model assumes no correlation between purchase frequency and monetary value, which is checked in the analysis.
5.  **CLV Calculation:** The predictions from the BG/NBD and Gamma-Gamma models are combined to calculate the Customer Lifetime Value for the desired 6-month and 12-month periods, incorporating a discount rate.
6.  **Customer Segmentation:** Based on the predicted 6-month CLV, customers are segmented into different value tiers (Hibernating, Need Attention, Loyal Customers, Champions) to facilitate targeted strategies.

## Code Description

The project consists of a single Python script (`Customer_Lifetime_Value.ipynb` - although presented as a notebook, it's structured as a Python script):

* **Import Libraries:** Imports necessary libraries including `lifetimes`, `pandas`, `numpy`, `datetime`, `matplotlib`, `seaborn`, and `sklearn`.
* **Read Data:** Loads the `Online_Retail.csv` dataset into a pandas DataFrame.
* **Data Understanding:** Provides initial exploration of the dataset using `head()`, `info()`, and `describe()`.
* **Data Preprocessing:**
    * Filters out transactions with negative quantities or unit prices.
    * Removes transactions with invoice numbers indicating returns (containing "C").
    * Handles missing `CustomerID` values by dropping rows.
    * Caps outliers in 'UnitPrice' and 'Quantity' using interquantile range.
    * Filters the data to include only transactions from the 'United Kingdom'.
    * Creates a 'Total Price' column by multiplying 'UnitPrice' and 'Quantity'.
* **Creating Summary Dataset:** Uses `lifetimes.utils.summary_data_from_transaction_data` to generate the RFM-style summary data for each customer. Customers with only one purchase are filtered out.
* **BG/NBD Model for Predicting Number of Purchase:**
    * Initializes and fits the `BetaGeoFitter` model to the summary data.
    * Provides a summary of the model coefficients.
    * Visualizes the frequency/recency matrix based on the fitted model.
    * Calculates the expected number of purchases in a 180-day (6-month) period for each customer.
* **Gamma - Gamma Model:**
    * Checks the correlation between 'frequency' and 'monetary_value' to validate the Gamma-Gamma model assumption.
    * Initializes and fits the `GammaGammaFitter` model.
* **6 months Customer Lifetime Value:**
    * Calculates the 6-month CLV for each customer using the fitted BG/NBD and Gamma-Gamma models, with a specified time period, frequency ('D' for daily), and discount rate (1%).
* **Segmentation Customers by 6 Months CLV:**
    * Segments customers into four tiers ('Hibernating', 'Need Attention', 'Loyal Customers', 'Champions') based on the quartiles of their predicted 6-month CLV.
* **Final Dataframe and Group by Segment:**
    * Displays the first few rows of the DataFrame with the CLV predictions and segments.
    * Calculates and displays the mean RFM and CLV values for each customer segment.

## Data Source

The project utilizes the `Online_Retail.csv` dataset, which contains transactional data from an online retail store.

## Libraries Used

* `lifetimes`: For probabilistic customer lifetime value models.
* `pandas`: For data manipulation and analysis.
* `numpy`: For numerical operations.
* `datetime`: For handling date and time information.
* `matplotlib`: For creating static, interactive, and animated visualizations in Python.
* `seaborn`: For making statistical graphics in Python.
* `sklearn.preprocessing.MinMaxScaler`: Although imported, it is not explicitly used in the provided code.

## Next Steps and Potential Improvements

* **12-Month CLV Prediction:** The commented-out code for predicting 12-month CLV can be implemented and analyzed.
* **Discount Rate Optimization:** Explore the impact of different discount rates on CLV.
* **Feature Engineering:** Investigate additional features that could improve the accuracy of the CLV predictions.
* **Model Evaluation:** Implement methods to evaluate the performance of the BG/NBD and Gamma-Gamma models.
* **Dynamic Segmentation:** Develop more dynamic customer segmentation strategies based on evolving CLV.
* **Integration with Business Systems:** Explore how the predicted CLV can be integrated into marketing automation or CRM systems for targeted actions.
* **Visualization:** Create more insightful visualizations of the CLV distribution and customer segments.

## Running the Code

To run the analysis, you will need:

1.  Python 3 installed on your system.
2.  The required libraries installed. You can install them using pip:
    ```bash
    pip install lifetimes pandas numpy matplotlib seaborn
    ```
3.  The `Online_Retail.csv` file in the same directory as the Python script or provide the correct path to the file.

Simply execute the `Customer_Lifetime_Value.ipynb` script (or run it as a Jupyter Notebook) to perform the CLV analysis and generate the results.
