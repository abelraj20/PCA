import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import seaborn as sns
from numpy.linalg import eig
from mpl_toolkits.mplot3d import Axes3D

data = load_iris()
df = pd.DataFrame(data.data, columns=data['feature_names'])
df['Species'] = data.target_names[data.target]
#df.head()

x = df["sepal length (cm)"]
y = df["sepal width (cm)"]

plt.plot(x,y, ".")
plt.show()

g = sns.pairplot(
    df,
    diag_kind = None,                  # Remove histograms from the diagonal
    hue='Species',
)

plt.savefig("pca1.png", bbox_inches='tight')
plt.show()

df = pd.DataFrame(data.data, columns=data['feature_names'])
fig, axes = plt.subplots(2, 2, figsize=(10, 6))

# Flatten the axes array for easier iteration
axes = axes.ravel()

for i, col in enumerate(df.columns):
    axes[i].hist(df[col], bins=20, color="white", edgecolor="black")
    axes[i].set_xlabel(f"{col}", color="black")               # Label x-axis
    axes[i].set_ylabel("Frequency", color="black")             # Label y-axis

# Adjust layout for better spacing
plt.tight_layout()

plt.savefig("pca2.png", bbox_inches='tight')
# Show the plot
plt.show()

mean = df.mean()

# Calculate the variance of each column
variance = df.var()

# Calculate the standard deviation of each column
std_dev = np.sqrt(variance)

# Standardise the DataFrame
df_s = (df - mean) / std_dev
print(df_s)

g = sns.pairplot(
    df_s,
    diag_kind = None,                  # Remove histograms from the diagonal
)

plt.savefig("pca3.png", bbox_inches='tight')
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(10, 6))

# Flatten the axes array for easier iteration
axes = axes.ravel()

for i, col in enumerate(df_s.columns):
    axes[i].hist(df_s[col], bins=20, color="white", edgecolor="black")
    axes[i].set_xlabel(f"{col}", color="black")               # Label x-axis
    axes[i].set_ylabel("Frequency", color="black")             # Label y-axis

# Adjust layout for better spacing
plt.tight_layout()

plt.savefig("pca4.png", bbox_inches='tight')
# Show the plot
plt.show()

def calculate_covariance_matrix(df):
    """Calculate the covariance matrix for a given DataFrame."""
    n = len(df)  # Number of data points
    means = df.mean()  # Mean of each column
    co_matrix = np.zeros((df.shape[1], df.shape[1]))  # Initialize covariance matrix

    # Loop over each pair of variables
    for i in range(df.shape[1]):
        for j in range(df.shape[1]):
            cov = np.sum((df.iloc[:, i] - means[i]) * (df.iloc[:, j] - means[j])) / (n)
            co_matrix[i, j] = cov

    return pd.DataFrame(co_matrix, index=df.columns, columns=df.columns)

# For unstandardised dataset
matrix = calculate_covariance_matrix(df)
print("Covariance Matrix (Unstandardised):")
print(matrix)

# Covariance matrix for the standardized dataset
matrix_s = calculate_covariance_matrix(df_s)
print("\nCovariance Matrix (Standardised):")
print(matrix_s)

eigenvalues, eigenvectors = eig(matrix_s)

# Display the results
print("Eigenvalues:")
print(eigenvalues)

print("\nEigenvectors:")
print(eigenvectors)

def load_data():
    """Load the Iris dataset."""
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['Species'] = data.target_names[data.target]  # Add species names
    return df

def standardise_df(df):
    """Standardize the DataFrame."""
    mean = df.mean()
    std_dev = df.std()  # Standard deviation
    df_s = (df - mean) / std_dev  # Standardization
    return df_s

def calculate_covariance_matrix(df):
    """Calculate the covariance matrix for a given DataFrame."""
    n = len(df)  # Number of data points
    means = df.mean()  # Mean of each column
    co_matrix = np.zeros((df.shape[1], df.shape[1]))  # Initialize covariance matrix

    # Loop over each pair of variables
    for i in range(df.shape[1]):
        for j in range(df.shape[1]):
            cov = np.sum((df.iloc[:, i] - means[i]) * (df.iloc[:, j] - means[j])) / n
            co_matrix[i, j] = cov

    return pd.DataFrame(co_matrix, index=df.columns, columns=df.columns)

def perform_pca(df):
    """Perform PCA on the given DataFrame."""
    # Standardize the DataFrame
    df_s = standardise_df(df.iloc[:, :-1])  # Exclude 'Species' column for PCA

    # Calculate the covariance matrix for the standardized dataset
    matrix_s = calculate_covariance_matrix(df_s)

    # Extract eigenvalues and eigenvectors
    eigenvalues, eigenvectors = eig(matrix_s)

    # Sort eigenvalues and eigenvectors
    sorted_indices = np.argsort(eigenvalues)[::-1]  # Sort in descending order
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # Project the standardized data onto the new eigenvector basis (principal components)
    PC = np.dot(df_s, sorted_eigenvectors)

    # Convert to DataFrame for easier interpretation
    PC_df = pd.DataFrame(PC, columns=[f"PC{i+1}" for i in range(PC.shape[1])])
    return PC_df, sorted_eigenvalues, sorted_eigenvectors

def main():
    df = load_data()

    # Perform PCA
    principal_components, eigenvalues, eigenvectors = perform_pca(df)

    # Ensure the 'Species' column is accessible
    principal_components['Species'] = df['Species'].values  # Add Species column to principal_components

    # Create the pairplot with custom colors
    g = sns.pairplot(
        principal_components,
        diag_kind=None,              # Remove histograms from the diagonal
        hue='Species',               # Color by species
        palette='muted',            # Color palette
    )

    plt.savefig("pca5.png", bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()

def load_data():
    """Load the Iris dataset."""
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['Species'] = data.target_names[data.target]  # Add species names
    return df

def standardise_df(df):
    """Standardize the DataFrame."""
    mean = df.mean()
    std_dev = df.std()  # Standard deviation
    df_s = (df - mean) / std_dev  # Standardization
    return df_s

def calculate_covariance_matrix(df):
    """Calculate the covariance matrix for a given DataFrame."""
    n = len(df)  # Number of data points
    means = df.mean()  # Mean of each column
    co_matrix = np.zeros((df.shape[1], df.shape[1]))  # Initialize covariance matrix

    # Loop over each pair of variables
    for i in range(df.shape[1]):
        for j in range(df.shape[1]):
            cov = np.sum((df.iloc[:, i] - means[i]) * (df.iloc[:, j] - means[j])) / n
            co_matrix[i, j] = cov

    return pd.DataFrame(co_matrix, index=df.columns, columns=df.columns)

def perform_pca(df):
    """Perform PCA on the given DataFrame."""
    # Standardize the DataFrame
    df_s = standardise_df(df.iloc[:, :-1])  # Exclude 'Species' column for PCA

    # Calculate the covariance matrix for the standardized dataset
    matrix_s = calculate_covariance_matrix(df_s)

    # Extract eigenvalues and eigenvectors
    eigenvalues, eigenvectors = eig(matrix_s)

    # Sort eigenvalues and eigenvectors
    sorted_indices = np.argsort(eigenvalues)[::-1]  # Sort in descending order
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # Project the standardized data onto the new eigenvector basis (principal components)
    PC = np.dot(df_s, sorted_eigenvectors)

    # Convert to DataFrame for easier interpretation
    PC_df = pd.DataFrame(PC, columns=[f"PC{i+1}" for i in range(PC.shape[1])])
    return PC_df, sorted_eigenvalues, sorted_eigenvectors

def main():
    # Load the data
    df = load_data()

    # Perform PCA
    PC_df, sorted_eigenvalues, sorted_eigenvectors = perform_pca(df)

    # Add the Species column to PC_df for plotting
    PC_df['Species'] = df['Species']

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Get unique species and assign muted colors
    unique_species = PC_df['Species'].unique()
    muted_palette = sns.color_palette("muted", n_colors=len(unique_species))  # Muted palette

    # Plot each species with a different color
    for i, species in enumerate(unique_species):
        species_data = PC_df[PC_df['Species'] == species]
        ax.scatter(species_data['PC1'], species_data['PC2'], species_data['PC3'],
                   c=[muted_palette[i]], label=species, marker='o')

    # Set labels
    ax.set_xlabel('PC1', labelpad=5)
    ax.set_ylabel('PC2', labelpad=5)
    ax.set_zlabel('PC3', labelpad=-2)

    # Change the viewing angle
    ax.view_init(elev=20, azim=30)  # Adjust these values to change the view angle

    # Show plot
    plt.tight_layout()  # Helps prevent cut-off labels
    plt.show()

if __name__ == "__main__":
    main()
