import pandas as pd
import pytz

class DataFrameTimeUtils:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    @property
    def index(self):
        return self.dataframe.index

    @index.setter
    def index(self, new_index):
        self.dataframe.index = new_index

    def convert_column_to_datetime(self, column_name):
        """Converts a column to datetime format in the DataFrame."""
        try:
            self.dataframe[column_name] = pd.to_datetime(self.dataframe[column_name])
        except KeyError:
            print(f"Column '{column_name}' does not exist in the DataFrame.")

    def add_timedelta_to_column(self, column_name, timedelta_value):
        """Adds a timedelta to a column in the DataFrame."""
        try:
            self.dataframe[column_name] += timedelta_value
        except KeyError:
            print(f"Column '{column_name}' does not exist in the DataFrame.")

    def subtract_timedelta_from_column(self, column_name, timedelta_value):
        """Subtracts a timedelta from a column in the DataFrame."""
        try:
            self.dataframe[column_name] -= timedelta_value
        except KeyError:
            print(f"Column '{column_name}' does not exist in the DataFrame.")

    def add_timedelta_to_index(self, timedelta_value):
        """Adds a timedelta to the index of the DataFrame."""
        if isinstance(self.dataframe.index, pd.DatetimeIndex):
            self.dataframe.index += timedelta_value
        else:
            print("Index is not in datetime format.")

    def subtract_timedelta_from_index(self, timedelta_value):
        """Subtracts a timedelta from the index of the DataFrame."""
        if isinstance(self.dataframe.index, pd.DatetimeIndex):
            self.dataframe.index -= timedelta_value
        else:
            print("Index is not in datetime format.")

    def create_date_list(self, start_date, end_date, granularity):
        """Creates a list of dates with the specified granularity."""
        date_range = pd.date_range(start=start_date, end=end_date, freq=granularity)
        return date_range.to_list()

    def convert_timezone(self, timezone):
        """Converts the DataFrame's index to the specified time zone."""
        if isinstance(self.dataframe.index, pd.DatetimeIndex):
            self.dataframe.index = self.dataframe.index.tz_convert(timezone)
        else:
            print("Index is not in datetime format.")

    def round_timestamps(self, freq):
        """Rounds the timestamps in the DataFrame's index to the specified frequency."""
        if isinstance(self.dataframe.index, pd.DatetimeIndex):
            self.dataframe.index = self.dataframe.index.round(freq)
        else:
            print("Index is not in datetime format.")

    def resample_dataframe(self, rule):
        """Resamples the DataFrame based on the specified rule."""
        if isinstance(self.dataframe.index, pd.DatetimeIndex):
            resampled_df = self.dataframe.resample(rule).mean()
            return resampled_df
        else:
            print("Index is not in datetime format.")

    def apply_time_shift(self, timedelta_value):
        """Applies a time shift to the DataFrame's index."""
        if isinstance(self.dataframe.index, pd.DatetimeIndex):
            self.dataframe.index = self.dataframe.index + timedelta_value
        else:
            print("Index is not in datetime format.")

    def __str__(self):
        return str(self.dataframe)

    def __repr__(self):
        return repr(self.dataframe)


if __name__ == "__main__":
    # Create a sample DataFrame
    df = pd.DataFrame({'Value': [1, 2, 3, 4]}, index=pd.date_range(start='2023-06-01', periods=4))

    # Create an instance of DataFrameTimeUtils
    time_utils = DataFrameTimeUtils(df)

    # Convert the DataFrame's index to a different time zone
    time_utils.convert_timezone('America/New_York')

    # Round the timestamps in the DataFrame's index to hours
    time_utils.round_timestamps('H')

    # Resample the DataFrame to daily frequency
    resampled_df = time_utils.resample_dataframe('D')

    # Apply a time shift of 3 hours to the DataFrame's index
    time_utils.apply_time_shift(pd.Timedelta(hours=3))

    # Print the original DataFrame
    print("Original DataFrame:")
    print(df)

    # Print the modified DataFrame
    print("\nModified DataFrame:")
    print(resampled_df)
