import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler


class DataAnalyzer:
    def __init__(self, data):
        self.data = data

    def __repr__(self):
        return f"DataAnalyzer(data={self.data})"

    @property
    def columns(self):
        """Get the column names of the data."""
        return self.data.columns.tolist()

    @staticmethod
    def _validate_data(data):
        """Validate if the input is a Pandas DataFrame."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Invalid data format. Ensure the input is a Pandas DataFrame.")

    def clean_data(self, columns=None):
        """
        Perform data cleaning operations by dropping duplicates and handling missing values.
        Args:
            columns (list or None): A list of column names to consider for cleaning.
            If None, all columns are considered. Default is None.
        Returns:
            A new Pandas DataFrame with the cleaned data.
        """
        cleaned_data = self.data.copy()

        # Drop duplicates
        cleaned_data.drop_duplicates(subset=columns, inplace=True)

        # Handle missing values
        if columns:
            cleaned_data.dropna(subset=columns, inplace=True)
        else:
            cleaned_data.dropna(inplace=True)

        return cleaned_data

    @staticmethod
    def remove_outliers(data, column, z_thresh=3):
        """
        Remove outliers from a column of numerical data in a Pandas DataFrame.
        Args:
            data (Pandas DataFrame): The DataFrame containing the data.
            column (str): The name of the column containing the data.
            z_thresh (float): The z-score threshold for identifying outliers. Default is 3.
        Returns:
            A new Pandas DataFrame with the outliers removed.
        """
        try:
            mean = np.mean(data[column])
            std = np.std(data[column])
            z_score = np.abs((data[column] - mean) / std)
            data = data[z_score <= z_thresh]
        except (KeyError, TypeError):
            print("Error: Invalid column name or non-numeric data in the specified column.")
            return None
        return data

    @staticmethod
    def impute_missing_values(data, column, method="median"):
        """
        Impute missing values in a column of numerical data in a Pandas DataFrame.
        Args:
            data (Pandas DataFrame): The DataFrame containing the data.
            column (str): The name of the column containing the data.
            method (str): The imputation method to use. Options are "median", "mean", and "mode". Default is "median".
        Returns:
            A new Pandas DataFrame with the missing values imputed.
        """
        try:
            if method == "median":
                imputed_value = data[column].median()
            elif method == "mean":
                imputed_value = data[column].mean()
            elif method == "mode":
                imputed_value = data[column].mode()[0]
            else:
                raise ValueError("Invalid imputation method")

            data[column] = data[column].fillna(imputed_value)
        except KeyError:
            print(f"Error: Invalid column name '{column}'")
            return None
        except TypeError:
            print(f"Error: Non-numeric data in column '{column}'")
            return None
        return data

    @staticmethod
    def load_data(file_path):
        """
        Load data from a file path into a Pandas DataFrame.
        Args:
            file_path (str): The path to the data file.
        Returns:
            A Pandas DataFrame containing the loaded data.
        """
        supported_formats = [".csv", ".xlsx", ".xls"]  # Add more supported formats if needed

        file_extension = file_path[file_path.rfind("."):].lower()

        if file_extension not in supported_formats:
            print("Error: unsupported file format")
            return None

        try:
            if file_extension == ".csv":
                data = pd.read_csv(file_path)
            elif file_extension in [".xlsx", ".xls"]:
                data = pd.read_excel(file_path)
            # Add more conditions for other file formats if needed
            else:
                print("Error: unsupported file format")
                return None
        except FileNotFoundError:
            print("Error: file not found")
            return None

        return data

    @staticmethod
    def encode_categorical_data(data, columns):
        """
        Encode categorical data in a Pandas DataFrame using one-hot encoding.
        Args:
            data (Pandas DataFrame): The DataFrame containing the data.
            columns (list): A list of column names containing the categorical data.
        Returns:
            A new Pandas DataFrame with the categorical data encoded using one-hot encoding.
        """
        try:
            encoded_data = data.copy()
            for col in columns:
                if col not in encoded_data.columns:
                    raise KeyError(f"Column '{col}' not found in the DataFrame")
                dummies = pd.get_dummies(encoded_data[col], prefix=col)
                encoded_data = pd.concat([encoded_data, dummies], axis=1)
                encoded_data = encoded_data.drop(columns=[col])
        except TypeError:
            print("Error: Invalid data format. Ensure the input is a Pandas DataFrame.")
            return None
        except KeyError as e:
            print(e)
            return None
        return encoded_data

    def normalize_data(self):
        """
        Perform data normalization using StandardScaler.
        Returns:
            A new Pandas DataFrame with the normalized data.
        """
        try:
            self._validate_data(self.data)
            scaler = StandardScaler()
            normalized_data = pd.DataFrame(scaler.fit_transform(self.data), columns=self.columns)
            return normalized_data
        except Exception as e:
            print(f"Error: {str(e)}")
            return None

    def filter_data(self, condition=None, columns=None):
        """
        Filter the data based on a given condition.
        Args:
            condition (str): A string representing the filtering condition. Default is None.
                Example conditions:
                - "age > 30"
                - "category == 'A'"
                - "price <= 100"
            columns (list): A list of column names to include in the filtered data. Default is None.
        Returns:
            A new Pandas DataFrame with the filtered data.
        """
        if not condition and not columns:
            raise ValueError("At least one of 'condition' or 'columns' must be provided.")

        filtered_data = self.data

        try:
            if condition:
                filtered_data = filtered_data.query(condition)
            if columns:
                filtered_data = filtered_data[columns]
        except Exception as e:
            print(f"Error: {str(e)}")
            return None

        return filtered_data

    def groupby(self, columns, aggregations):
        """
        Perform groupby operation on the data.
        Args:
            columns (list): A list of column names to group by.
            aggregations (dict): A dictionary specifying the column names as keys and
            the corresponding aggregation functions as values.
                Example aggregations:
                {
                    'column_name1': 'mean',
                    'column_name2': ['sum', 'max'],
                    'column_name3': lambda x: x.nunique()
                }
        Returns:
            A new Pandas DataFrame with the grouped and aggregated data.
        """
        if not columns or not aggregations:
            raise ValueError("Both 'columns' and 'aggregations' must be provided.")

        grouped_data = self.data

        try:
            grouped_data = grouped_data.groupby(columns).agg(aggregations)
        except Exception as e:
            print(f"Error: {str(e)}")
            return None

        return grouped_data

    def merge_data(self, other_data, on=None, how='inner'):
        """
        Merge the data with another DataFrame.
        Args:
            other_data (Pandas DataFrame): The DataFrame to merge with.
            on (str or list): Column names to join on. If None, it merges on all common columns.
                Example usage:
                - Single column: 'column_name'
                - Multiple columns: ['column_name1', 'column_name2']
            how (str): The type of merge to perform. Options are 'inner', 'left', 'right', 'outer'. Default is 'inner'.
        Returns:
            A new Pandas DataFrame with the merged data.
        """
        if not isinstance(other_data, pd.DataFrame):
            raise ValueError("'other_data' must be a Pandas DataFrame.")

        merged_data = self.data

        try:
            merged_data = pd.merge(merged_data, other_data, on=on, how=how)
        except Exception as e:
            print(f"Error: {str(e)}")
            return None

        return merged_data

    def save_data(self, file_path, file_format='csv'):
        """
        Save the data to a file.
        Args:
            file_path (str): The path to save the file.
            file_format (str): The format of the file to save. Options are 'csv', 'xlsx', and 'xls'. Default is 'csv'.
        Raises:
            ValueError: If an unsupported file format is provided.
        """
        supported_formats = ['csv', 'xlsx', 'xls']

        if file_format not in supported_formats:
            raise ValueError("Unsupported file format")

        try:
            if file_format == 'csv':
                self.data.to_csv(file_path, index=False)
            elif file_format == 'xlsx' or file_format == 'xls':
                self.data.to_excel(file_path, index=False)
        except Exception as e:
            print(f"Error: {str(e)}")

    def describe_data(self):
        """
        Generate descriptive statistics for the data.
        Returns:
            A Pandas DataFrame containing the descriptive statistics.
        """
        try:
            description = self.data.describe()
            return description
        except Exception as e:
            print(f"Error: {str(e)}")
            return None


if __name__ == "__main__":
    # Example 1: Loading data from a file and performing data analysis
    file_path = "data.csv"
    analyzer = DataAnalyzer.load_data(file_path)
    if analyzer:
        print(f"Loaded data:\n{analyzer.data.head()}")

        # Perform data cleaning
        cleaned_data = analyzer.clean_data()
        print(f"Cleaned data:\n{cleaned_data.head()}")

        # Remove outliers
        outlier_removed_data = DataAnalyzer.remove_outliers(cleaned_data, "column_name")
        print(f"Data with outliers removed:\n{outlier_removed_data.head()}")

        # Impute missing values
        imputed_data = DataAnalyzer.impute_missing_values(outlier_removed_data, "column_name")
        print(f"Data with missing values imputed:\n{imputed_data.head()}")

        # Perform data normalization
        normalized_data = analyzer.normalize_data()
        print(f"Normalized data:\n{normalized_data.head()}")

        # Save the cleaned and normalized data to a file
        analyzer.save_data("cleaned_normalized_data.csv")

        # Generate descriptive statistics
        description = analyzer.describe_data()
        print(f"Descriptive statistics:\n{description}")

    # Example 2: Filtering and grouping data
    data = pd.DataFrame({"column1": [1, 2, 3, 4, 5],
                         "column2": ["A", "B", "A", "B", "A"],
                         "column3": [10, 20, 30, 40, 50]})
    analyzer = DataAnalyzer(data)

    # Filter the data based on a condition
    filtered_data = analyzer.filter_data(condition="column1 > 3")
    print(f"Filtered data:\n{filtered_data}")

    # Group the data and compute aggregations
    columns_to_groupby = ["column2"]
    aggregations = {"column1": "sum", "column3": ["mean", "max"]}
    grouped_data = analyzer.groupby(columns_to_groupby, aggregations)
    print(f"Grouped data:\n{grouped_data}")

    # Example 3: Merging data
    data1 = pd.DataFrame({"column1": [1, 2, 3], "column2": ["A", "B", "C"]})
    data2 = pd.DataFrame({"column1": [4, 5, 6], "column2": ["D", "E", "F"]})
    analyzer1 = DataAnalyzer(data1)
    merged_data = analyzer1.merge_data(data2, on="column1", how="outer")
    print(f"Merged data:\n{merged_data}")
