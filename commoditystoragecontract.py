import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def price_storage_contract(
    injection_dates, withdrawal_dates,
    injection_rate, withdrawal_rate,
    max_volume, storage_cost_per_unit, 
    all_data
):
    """
    Prototype pricing model with visualization.
    """
    
    volume = 0
    cashflows = []
    
    # Injection phase
    for d in injection_dates:
        date = pd.to_datetime(d)
        price = np.interp(
            date.value,
            all_data.index.view(np.int64),
            all_data.values
        )
        inject_amount = min(injection_rate, max_volume - volume)
        cost = inject_amount * price
        volume += inject_amount
        storage_cost = volume * storage_cost_per_unit
        cashflows.append({"Date": date, "Action": "Injection", 
                          "Amount": inject_amount, "Price": price, 
                          "Cashflow": -cost - storage_cost, "Volume": volume})
    
    # Withdrawal phase
    for d in withdrawal_dates:
        date = pd.to_datetime(d)
        price = np.interp(
            date.value,
            all_data.index.view(np.int64),
            all_data.values
        )
        withdraw_amount = min(withdrawal_rate, volume)
        revenue = withdraw_amount * price
        volume -= withdraw_amount
        storage_cost = volume * storage_cost_per_unit
        cashflows.append({"Date": date, "Action": "Withdrawal", 
                          "Amount": withdraw_amount, "Price": price, 
                          "Cashflow": revenue - storage_cost, "Volume": volume})
    
    # Prepare cashflow dataframe
    df_cashflows = pd.DataFrame(cashflows).sort_values("Date")
    total_value = df_cashflows["Cashflow"].sum()
    
    # -------------------------
    # Plot Price & Volume
    # -------------------------
    fig, ax1 = plt.subplots(figsize=(10,5))
    
    # Price curve
    ax1.plot(all_data.index, all_data.values, label="Gas Price", color='blue')
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Mark injections & withdrawals
    ax1.scatter(df_cashflows[df_cashflows['Action']=='Injection']['Date'],
                df_cashflows[df_cashflows['Action']=='Injection']['Price'],
                marker='^', color='green', s=100, label="Injection")
    ax1.scatter(df_cashflows[df_cashflows['Action']=='Withdrawal']['Date'],
                df_cashflows[df_cashflows['Action']=='Withdrawal']['Price'],
                marker='v', color='red', s=100, label="Withdrawal")
    
    # Volume curve on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(df_cashflows['Date'], df_cashflows['Volume'], label="Stored Volume", color='orange', linestyle='--')
    ax2.set_ylabel("Volume", color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')
    
    fig.tight_layout()
    fig.legend(loc="upper left", bbox_to_anchor=(0.1,0.9))
    plt.title("Gas Storage Contract Visualization")
    plt.show()
    
    return total_value, df_cashflows


# -------------------------
# Test Example
# -------------------------
dates = pd.date_range("2024-01-31", periods=12, freq="M")
prices = np.linspace(20, 40, 12)
all_data = pd.Series(prices, index=dates)

injection_dates = ["2024-02-28", "2024-03-31", "2024-04-30"]
withdrawal_dates = ["2024-08-31", "2024-09-30"]
injection_rate = 100
withdrawal_rate = 150
max_volume = 500
storage_cost_per_unit = 0.2

value, cashflows = price_storage_contract(
    injection_dates, withdrawal_dates,
    injection_rate, withdrawal_rate,
    max_volume, storage_cost_per_unit,
    all_data
)

print("Total Contract Value:", round(value,2))
print("\nCashflow Breakdown:\n", cashflows)
