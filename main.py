import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# -------------------------------------------------------------
# MODULE 2: Warehouse Data Management & Feature Engineering
# -------------------------------------------------------------
def load_and_preprocess_warehouse_data():
    # Large synthetic warehouse data generation (200 rows across 4 zones)
    np.random.seed(42)
    n = 200
    zones = ['Z1', 'Z2', 'Z3', 'Z4']
    zone_ids = [zones[i % 4] for i in range(n)]
    shelf_ids = [f"S{(i // 4 + 1):03d}{i % 4 + 1}" for i in range(n)]

    total_capacity  = np.random.randint(200, 1000, size=n)
    # Occupied space: randomly between 5% and 100% of total capacity
    occupied_pct    = np.random.uniform(0.05, 1.00, size=n)
    occupied_space  = (total_capacity * occupied_pct).astype(int)
    stock_inflow    = np.random.randint(5, 100, size=n)
    stock_outflow   = np.random.randint(1, stock_inflow)  # always <= inflow

    data = {
        'Zone_ID':        zone_ids,
        'Shelf_ID':       shelf_ids,
        'Total_Capacity': total_capacity,
        'Occupied_Space': occupied_space,
        'Stock_Inflow':   stock_inflow,
        'Stock_Outflow':  stock_outflow,
    }
    
    df = pd.DataFrame(data)
    
    # Feature Engineering: Calculating utilization percentage and available capacity[cite: 2]
    df['Available_Capacity'] = df['Total_Capacity'] - df['Occupied_Space']
    df['Utilization_Pct'] = (df['Occupied_Space'] / df['Total_Capacity']) * 100
    df['Stock_Turnover'] = df['Stock_Inflow'] - df['Stock_Outflow']
    
    # Categorizing utilization levels (Low, Optimal, Overutilized)[cite: 2]
    def label_utilization(pct):
        if pct < 50:
            return 'Low'
        elif 50 <= pct <= 85:
            return 'Optimal'
        else:
            return 'Overutilized'
            
    df['Utilization_Status'] = df['Utilization_Pct'].apply(label_utilization)
    return df

# -------------------------------------------------------------
# MODULE 3: Space Utilization Analysis (Unsupervised Learning - Clustering)
# -------------------------------------------------------------
def apply_clustering(df):
    print("--- Unsupervised Learning: Clustering Zones/Shelves ---")
    X = df[['Total_Capacity', 'Occupied_Space', 'Utilization_Pct']]
    
    # Applying KMeans clustering to group similar utilization behavior[cite: 2]
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X)
    print(df[['Shelf_ID', 'Utilization_Pct', 'Cluster']])
    return df

# -------------------------------------------------------------
# MODULE 3: Space Utilization Analysis (Supervised Learning - Classification)
# -------------------------------------------------------------
def apply_supervised_learning(df):
    print("\n--- Supervised Learning: Predicting Utilization Status ---")
    X = df[['Total_Capacity', 'Occupied_Space', 'Stock_Inflow', 'Stock_Outflow', 'Stock_Turnover']]
    y = df['Utilization_Status']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # Using Random Forest Classifier to label locations as low, optimal, or overutilized[cite: 2]
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

# -------------------------------------------------------------
# Main Execution Pipeline
# -------------------------------------------------------------
if __name__ == "__main__":
    warehouse_df = load_and_preprocess_warehouse_data()
    print("Processed Warehouse Data with Features:")
    print(warehouse_df.head(), "\n")
    
    clustered_df = apply_clustering(warehouse_df)
    apply_supervised_learning(clustered_df)