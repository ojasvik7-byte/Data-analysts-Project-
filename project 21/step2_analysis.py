"""
PROJECT 21 - Online Retail Customer Purchase Analysis
STEP 2 - Run the analysis (all 15 tasks)
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                    # lets charts be saved without a window
import matplotlib.pyplot as plt

pd.set_option("display.width", 120)


def heading(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


# ---------------------------------------------------------------- TASK 1
heading("TASK 1: Load and inspect")
df = pd.read_csv("21_retail_customer_orders.csv")
print("Rows, columns:", df.shape)
print("Column names :", list(df.columns))
print("\nData types:\n", df.dtypes)
print("\nFirst 5 rows:\n", df.head())

# ---------------------------------------------------------------- TASK 2
heading("TASK 2: Missing values and duplicate Order_IDs")
print("Missing values per column:\n", df.isnull().sum())
print("\nDuplicate Order_ID rows:", df["Order_ID"].duplicated().sum())
df = df.drop_duplicates(subset="Order_ID", keep="first")
print("Rows after removing duplicates:", len(df))

# ---------------------------------------------------------------- TASK 3
heading("TASK 3: Handle missing Quantity / Unit_Price")
# Unit_Price: a product always has the same price, so we fill using the
# median price of the SAME product (the most accurate guess).
df["Unit_Price"] = df["Unit_Price"].fillna(
    df.groupby("Product")["Unit_Price"].transform("median"))
# Quantity: most orders are small and a few are very large, so the MEDIAN
# is safer than the mean (the mean gets pulled up by bulk orders).
df["Quantity"] = df["Quantity"].fillna(df["Quantity"].median())
print("Missing values left:\n", df.isnull().sum())

# ---------------------------------------------------------------- TASK 4
heading("TASK 4: Convert Order_Date and extract parts")
df["Order_Date"] = pd.to_datetime(df["Order_Date"])
df["Year"] = df["Order_Date"].dt.year
df["Month"] = df["Order_Date"].dt.month
df["Month_Name"] = df["Order_Date"].dt.month_name()
df["Weekday"] = df["Order_Date"].dt.day_name()
print(df[["Order_Date", "Year", "Month", "Month_Name", "Weekday"]].head())

# ---------------------------------------------------------------- TASK 6
# (done before Task 5 because Task 5 needs Order_Value for revenue)
df["Order_Value"] = df["Quantity"] * df["Unit_Price"]

# ---------------------------------------------------------------- TASK 5
heading("TASK 5: Totals")
total_orders = df["Order_ID"].nunique()
total_units = int(df["Quantity"].sum())
total_revenue = df["Order_Value"].sum()
print("Total orders       :", total_orders)
print("Total units sold   :", total_units)
print(f"Total revenue      : Rs {total_revenue:,.0f}")

heading("TASK 6: Order_Value column created")
print(df[["Order_ID", "Quantity", "Unit_Price", "Order_Value"]].head())

# ---------------------------------------------------------------- TASK 7
heading("TASK 7: Top 10 products by revenue")
top_products = (df.groupby("Product")["Order_Value"].sum()
                  .sort_values(ascending=False).head(10))
print(top_products)

# ---------------------------------------------------------------- TASK 8
heading("TASK 8: Categories ranked by quantity and revenue")
cat_summary = (df.groupby("Category")
                 .agg(Total_Quantity=("Quantity", "sum"),
                      Total_Revenue=("Order_Value", "sum"))
                 .sort_values("Total_Revenue", ascending=False))
cat_summary["Qty_Rank"] = cat_summary["Total_Quantity"].rank(ascending=False).astype(int)
cat_summary["Revenue_Rank"] = cat_summary["Total_Revenue"].rank(ascending=False).astype(int)
print(cat_summary)

# ---------------------------------------------------------------- TASK 9
heading("TASK 9: Average order value by payment method")
print(df.groupby("Payment_Method")["Order_Value"].mean()
        .sort_values(ascending=False).round(2))

# ---------------------------------------------------------------- TASK 10
heading("TASK 10: Busiest days and peak month")
print("Orders per weekday:\n", df["Weekday"].value_counts())
monthly = df.groupby(["Month", "Month_Name"])["Order_Value"].sum().reset_index()
peak = monthly.loc[monthly["Order_Value"].idxmax()]
print(f"\nPeak sales month: {peak['Month_Name']} (Rs {peak['Order_Value']:,.0f})")
print("\nMonthly sales trend:\n", monthly.to_string(index=False))

# Chart for the monthly trend
plt.figure(figsize=(9, 4))
plt.plot(monthly["Month_Name"].str[:3], monthly["Order_Value"], marker="o")
plt.title("Monthly Sales Trend (2025)")
plt.xlabel("Month")
plt.ylabel("Revenue (Rs)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("monthly_sales_trend.png", dpi=120)
plt.close()

# ---------------------------------------------------------------- TASK 11
heading("TASK 11: Dictionary and set")
sales_by_category = df.groupby("Category")["Order_Value"].sum().round(0).to_dict()
print("Dictionary (sales by category):")
for cat, val in sales_by_category.items():
    print(f"  {cat:<12} Rs {val:,.0f}")
unique_categories = set(df["Category"])
print("\nSet of unique categories:", unique_categories)

# ---------------------------------------------------------------- TASK 12
heading("TASK 12: Function - total sales of a category")


def category_sales(category_name, data):
    """Return total sales for one category (0 if the category does not exist)."""
    subset = data[data["Category"] == category_name]
    return subset["Order_Value"].sum()


print("Electronics sales:", f"Rs {category_sales('Electronics', df):,.0f}")
print("Home sales       :", f"Rs {category_sales('Home', df):,.0f}")
print("Toys (not found) :", category_sales("Toys", df))

# ---------------------------------------------------------------- TASK 13
heading("TASK 13: NumPy statistics of Order_Value")
values = df["Order_Value"].to_numpy()
print(f"Mean              : {np.mean(values):,.2f}")
print(f"Median            : {np.median(values):,.2f}")
print(f"Standard deviation: {np.std(values):,.2f}")

# ---------------------------------------------------------------- TASK 14
heading("TASK 14: IQR outliers and order-value bands")
q1, q3 = np.percentile(values, [25, 75])
iqr = q3 - q1
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
print(f"Q1={q1:,.0f}  Q3={q3:,.0f}  IQR={iqr:,.0f}")
print(f"Outlier limits: below {lower:,.0f} or above {upper:,.0f}")
outliers = df[(df["Order_Value"] < lower) | (df["Order_Value"] > upper)]
print(f"\nNumber of unusually high orders: {len(outliers)}")
print(outliers[["Order_ID", "Product", "Quantity", "Order_Value"]]
      .sort_values("Order_Value", ascending=False).to_string(index=False))

# Bands: Low = bottom third, Medium = middle third, High = top third
df["Value_Band"] = pd.qcut(df["Order_Value"], q=3, labels=["Low", "Medium", "High"])
print("\nOrder-value bands (equal-sized thirds):\n", df["Value_Band"].value_counts())

# ---------------------------------------------------------------- TASK 15
heading("TASK 15: Product class and recommendations")


class Product:
    def __init__(self, name, price):
        self.name = name
        self.price = price

    def show(self):
        return f"{self.name} - Rs {self.price:,.0f}"


best_name = top_products.index[0]
best_price = df.loc[df["Product"] == best_name, "Unit_Price"].iloc[0]
best = Product(best_name, best_price)
print("Best-selling product object:", best.show())

best_cat = cat_summary.index[0]
weak_cat = cat_summary.index[-1]
print(f"""
FIVE RECOMMENDATIONS (based on the results above)
1. Keep '{best_name}' and other top-10 products always in stock and feature
   them on the home page - they bring in the most revenue.
2. '{best_cat}' is the top-earning category: run bundle deals inside it.
3. '{weak_cat}' earns the least: test discounts or better product photos.
4. Peak sales came in {peak['Month_Name']}: plan stock and ads before that month.
5. The {len(outliers)} unusually high-value orders (many are bulk
   purchases) matter a lot: offer wholesale pricing or loyalty rewards to
   those customers to encourage repeat big orders.
""")

# ---------------------------------------------------------------- SAVE
df.to_csv("cleaned_retail_orders.csv", index=False)
print("Saved: cleaned_retail_orders.csv and monthly_sales_trend.png")
