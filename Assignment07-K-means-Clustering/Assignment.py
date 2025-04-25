# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv(r'C:\Users\SUDARSHAN\Desktop\ML-LAB-120\K-means-Clustering\clustering.csv')

print("First 5 rows of dataset:")
print(data.head())

X = data[["LoanAmount", "ApplicantIncome"]]

plt.scatter(X["ApplicantIncome"], X["LoanAmount"], c='black')
plt.xlabel('Annual Income')
plt.ylabel('Loan Amount (In Thousands)')
plt.title('Initial Data')
plt.show()


K = 3


Centroids = X.sample(n=K)

# Visualize initial centroids
plt.scatter(X["ApplicantIncome"], X["LoanAmount"], c='black', label='Data Points')
plt.scatter(Centroids["ApplicantIncome"], Centroids["LoanAmount"], c='red', label='Initial Centroids')
plt.xlabel('Annual Income')
plt.ylabel('Loan Amount (In Thousands)')
plt.legend()
plt.title('Initial Centroids')
plt.show()

diff = 1
j = 0

while(diff != 0):
    XD = X.copy()
    i = 1
    # Step 3: Calculate Euclidean distance to each centroid
    for index1, row_c in Centroids.iterrows():
        ED = []
        for index2, row_d in XD.iterrows():
            d1 = (row_c["ApplicantIncome"] - row_d["ApplicantIncome"])**2
            d2 = (row_c["LoanAmount"] - row_d["LoanAmount"])**2
            d = np.sqrt(d1 + d2)
            ED.append(d)
        X[i] = ED
        i += 1


    C = []
    for index, row in X.iterrows():
        min_dist = row[1]
        pos = 1
        for i in range(K):
            if row[i+1] < min_dist:
                min_dist = row[i+1]
                pos = i + 1
        C.append(pos)
    X["Cluster"] = C

    
    Centroids_new = X.groupby(["Cluster"]).mean()[["LoanAmount", "ApplicantIncome"]]
    
    if j == 0:
        diff = 1
        j += 1
    else:
        diff = (Centroids_new['LoanAmount'] - Centroids['LoanAmount']).sum() + \
               (Centroids_new['ApplicantIncome'] - Centroids['ApplicantIncome']).sum()
        print("Centroid difference:", diff)

    Centroids = Centroids_new

# Final Cluster Visualization
color = ['blue', 'green', 'cyan']
for k in range(K):
    data_k = X[X["Cluster"] == k+1]
    plt.scatter(data_k["ApplicantIncome"], data_k["LoanAmount"], c=color[k], label=f'Cluster {k+1}')
plt.scatter(Centroids["ApplicantIncome"], Centroids["LoanAmount"], c='red', label='Centroids', marker='X', s=200)
plt.xlabel('Annual Income')
plt.ylabel('Loan Amount (In Thousands)')
plt.title('K-Means Clustering Output')
plt.legend()
plt.show()
