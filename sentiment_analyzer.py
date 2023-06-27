import pandas as pd
import numpy as np
from textblob import TextBlob
from textblob.exceptions import TextBlobException
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


class SentimentAnalyzer:
    def __init__(self):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None

    def load_data(self, data_source):
        """
        Load data from a given data source into a Pandas DataFrame.

        Args:
            data_source (str): Path or URL of the data source.

        Returns:
            bool: True if data is successfully loaded, False otherwise.
        """
        try:
            if data_source.endswith('.csv'):
                self.data = pd.read_csv(data_source)
            elif data_source.endswith('.xlsx'):
                self.data = pd.read_excel(data_source)
            else:
                raise ValueError("Unsupported data format.")

            # Additional data validation checks can be performed here

            return True

        except FileNotFoundError:
            print(f"Failed to load data from {data_source}: File not found.")
            return False

        except pd.errors.ParserError as e:
            print(f"Failed to load data from {data_source}: Error parsing the file.")
            print(f"Error details: {str(e)}")
            return False

        except Exception as e:
            print(f"An error occurred during data loading: {str(e)}")
            return False

    def analyze_sentiment(self, text_column, output_column):
        """
        Perform sentiment analysis on the text column of the loaded data.

        Args:
            text_column (str): Name of the column containing the text data.
            output_column (str): Name of the column to store sentiment analysis results.

        Returns:
            bool: True if sentiment analysis is successful, False otherwise.
        """
        if self.data is None:
            print("No data loaded. Please load data before performing sentiment analysis.")
            return False

        if text_column not in self.data.columns:
            print(f"Failed to analyze sentiment. Column '{text_column}' does not exist in the data.")
            return False

        try:
            sentiments = self.data[text_column].apply(lambda text: TextBlob(text).sentiment.polarity)
            self.data[output_column] = sentiments
            return True

        except TextBlobException as e:
            print(f"An error occurred during sentiment analysis: {str(e)}")
            return False

        except Exception as e:
            print(f"An error occurred during sentiment analysis: {str(e)}")
            return False

    def get_data(self):
        """
        Get the loaded data.

        Returns:
            pandas.DataFrame: The loaded data.
        """
        return self.data

    def preprocess_text(self, text_column, preprocessed_column):
        """
        Preprocess the text in a specified column by removing stopwords and special characters.

        Args:
            text_column (str): Name of the column containing the text data.
            preprocessed_column (str): Name of the column to store preprocessed text.

        Returns:
            bool: True if preprocessing is successful, False otherwise.
        """
        if self.data is None:
            print("No data loaded. Please load data before preprocessing text.")
            return False

        if text_column not in self.data.columns:
            print(f"Failed to preprocess text. Column '{text_column}' does not exist in the data.")
            return False

        try:
            # Perform preprocessing operations (e.g., removing stopwords, special characters)
            preprocessed_texts = self.data[text_column].apply(self._custom_preprocessing)
            self.data[preprocessed_column] = preprocessed_texts
            return True

        except Exception as e:
            print(f"An error occurred during text preprocessing: {str(e)}")
            return False

    def generate_word_frequency(self, text_column):
        """
        Generate word frequency statistics from a specified text column.

        Args:
            text_column (str): Name of the column containing the text data.

        Returns:
            pandas.DataFrame: DataFrame with word frequency statistics.
        """
        if self.data is None:
            print("No data loaded. Please load data before generating word frequency statistics.")
            return None

        if text_column not in self.data.columns:
            print(f"Failed to generate word frequency. Column '{text_column}' does not exist in the data.")
            return None

        try:
            text_corpus = self.data[text_column].tolist()

            # Create a CountVectorizer and fit-transform the text corpus
            count_vectorizer = CountVectorizer()
            word_counts = count_vectorizer.fit_transform(text_corpus)

            # Get the feature names (words)
            words = count_vectorizer.get_feature_names()

            # Compute the sum of word occurrences across the corpus
            word_frequencies = word_counts.sum(axis=0).tolist()[0]

            # Create a DataFrame with word frequency statistics
            word_stats = pd.DataFrame({'Word': words, 'Frequency': word_frequencies})

            return word_stats

        except Exception as e:
            print(f"An error occurred during word frequency generation: {str(e)}")
            return None

    def save_data(self, output_file):
        """
        Save the analyzed data to a file.

        Args:
            output_file (str): Path of the output file.

        Returns:
            bool: True if data is successfully saved, False otherwise.
        """
        if self.data is None:
            print("No data available to save.")
            return False

        try:
            self.data.to_csv(output_file, index=False)
            print(f"Data saved to '{output_file}'")
            return True

        except Exception as e:
            print(f"An error occurred during data saving: {str(e)}")
            return False

    @staticmethod
    def _custom_preprocessing(text):
        """
        Apply custom preprocessing steps to a single text.

        Args:
            text (str): Input text to preprocess.

        Returns:
            str: Preprocessed text.
        """
        # Implement your custom text preprocessing steps here
        # For example: remove stopwords, special characters, perform stemming/lemmatization, etc.
        preprocessed_text = text.lower()  # Placeholder example (convert text to lowercase)
        return preprocessed_text

    def visualize_sentiment(self, sentiment_column):
        """
        Visualize sentiment analysis results using a bar plot.

        Args:
            sentiment_column (str): Name of the column containing sentiment analysis results.

        Returns:
            bool: True if visualization is successful, False otherwise.
        """
        if self.data is None:
            print("No data loaded. Please load data before visualizing sentiment analysis.")
            return False

        if sentiment_column not in self.data.columns:
            print(f"Failed to visualize sentiment. Column '{sentiment_column}' does not exist in the data.")
            return False

        try:
            sentiment_counts = self.data[sentiment_column].value_counts()
            sentiment_counts.plot(kind='bar')
            plt.xlabel('Sentiment')
            plt.ylabel('Count')
            plt.title('Sentiment Analysis')
            plt.show()
            return True

        except Exception as e:
            print(f"An error occurred during sentiment visualization: {str(e)}")
            return False

    def export_data(self, output_format):
        """
        Export the analyzed data to a specified output format.

        Args:
            output_format (str): Output format (e.g., 'csv', 'json').

        Returns:
            bool: True if data is successfully exported, False otherwise.
        """
        if self.data is None:
            print("No data available to export.")
            return False

        try:
            if output_format == 'csv':
                self.data.to_csv('output.csv', index=False)
                print("Data exported to CSV successfully.")
                return True
            elif output_format == 'json':
                self.data.to_json('output.json', orient='records')
                print("Data exported to JSON successfully.")
                return True
            else:
                print("Invalid output format specified.")
                return False

        except Exception as e:
            print(f"An error occurred during data export: {str(e)}")
            return False

    def split_data(self, target_column, test_size=0.2, random_state=None):
        """
        Split the loaded data into training and testing sets.

        Args:
            target_column (str): Name of the column containing the data.
            test_size (float): The proportion of the data to include in the test split.
            random_state (int): The seed used by the random number generator.

        Returns:
            bool: True if data split is successful, False otherwise.
        """
        if self.data is None:
            print("No data loaded. Please load data before splitting.")
            return False

        try:
            X = self.data.drop(columns=[target_column])
            y = self.data[target_column]
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            return True

        except Exception as e:
            print(f"An error occurred during data split: {str(e)}")
            return False

    def train_model(self):
        """
        Train a sentiment analysis model using logistic regression.

        Returns:
            bool: True if model training is successful, False otherwise.
        """
        if self.X_train is None or self.y_train is None:
            print("No training data available. Please split the data and perform training.")
            return False

        try:
            self.model = LogisticRegression()
            self.model.fit(self.X_train, self.y_train)
            return True

        except Exception as e:
            print(f"An error occurred during model training: {str(e)}")
            return False

    def predict_sentiment(self, text):
        """
        Predict the sentiment of a given text using the trained model.

        Args:
            text (str): The text to predict the sentiment for.

        Returns:
            str: The predicted sentiment label.
        """
        if self.model is None:
            print("No model available. Please train a model before making predictions.")
            return None

        try:
            prediction = self.model.predict([text])
            return prediction[0]

        except Exception as e:
            print(f"An error occurred during sentiment prediction: {str(e)}")
            return None

    def evaluate_model(self):
        """
        Evaluate the performance of the trained model on the test set.

        Returns:
            float: The accuracy score of the model on the test set.
        """
        if self.X_test is None or self.y_test is None:
            print("No test data available. Please split the data and perform evaluation.")
            return None

        try:
            y_pred = self.model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            return accuracy

        except Exception as e:
            print(f"An error occurred during model evaluation: {str(e)}")
            return None

    def plot_word_frequency(self, text_column):
        """
        Plot word frequency distribution using a histogram.

        Args:
            text_column (str): Name of the column containing the text data.

        Returns:
            bool: True if plotting is successful, False otherwise.
        """
        if self.data is None:
            print("No data loaded. Please load data before plotting word frequency.")
            return False

        if text_column not in self.data.columns:
            print(f"Failed to plot word frequency. Column '{text_column}' does not exist in the data.")
            return False

        try:
            word_counts = self.data[text_column].str.split(expand=True).stack().value_counts()
            sns.histplot(data=word_counts, stat="frequency")
            plt.xlabel('Word')
            plt.ylabel('Frequency')
            plt.title('Word Frequency Distribution')
            plt.show()
            return True

        except Exception as e:
            print(f"An error occurred during word frequency plotting: {str(e)}")
            return False

    def get_top_n_words(self, text_column, n=10):
        """
        Get the top N most frequent words in a text column.

        Args:
            text_column (str): Name of the column containing the text data.
            n (int): Number of top words to retrieve (default: 10).

        Returns:
            list: List of tuples containing the top N words and their frequencies.
        """
        if self.data is None:
            print("No data loaded. Please load data before retrieving top words.")
            return []

        if text_column not in self.data.columns:
            print(f"Failed to retrieve top words. Column '{text_column}' does not exist in the data.")
            return []

        try:
            word_counts = self.data[text_column].str.split(expand=True).stack().value_counts()
            top_words = word_counts.head(n).reset_index().values.tolist()
            return top_words

        except Exception as e:
            print(f"An error occurred while retrieving top words: {str(e)}")
            return []

    def save_model(self, file_path):
        """
        Save the trained model to a file.

        Args:
            file_path (str): Path to save the model file.

        Returns:
            bool: True if model is successfully saved, False otherwise.
        """
        if self.model is None:
            print("No model available. Please train a model before saving.")
            return False

        try:
            joblib.dump(self.model, file_path)
            print("Model saved successfully.")
            return True

        except Exception as e:
            print(f"An error occurred while saving the model: {str(e)}")
            return False

    def filter_by_sentiment(self, sentiment_column, sentiment_threshold):
        """
        Filter the data based on a sentiment threshold.

        Args:
            sentiment_column (str): Name of the column containing the sentiment scores.
            sentiment_threshold (float): Threshold value to filter the data.

        Returns:
            pandas.DataFrame: Filtered data based on the sentiment threshold.
        """
        if self.data is None:
            print("No data loaded. Please load data before filtering by sentiment.")
            return None

        if sentiment_column not in self.data.columns:
            print(f"Failed to filter data by sentiment. Column '{sentiment_column}' does not exist in the data.")
            return None

        try:
            filtered_data = self.data[self.data[sentiment_column] >= sentiment_threshold]
            return filtered_data

        except Exception as e:
            print(f"An error occurred while filtering data by sentiment: {str(e)}")
            return None

    def remove_duplicates(self, subset=None, keep='first'):
        """
        Remove duplicate rows from the loaded data.

        Args:
            subset (str or list): Columns to consider for identifying duplicates (default: None, consider all columns).
            keep (str): Strategy to keep duplicates ('first', 'last', 'False', or 'None', default: 'first').

        Returns:
            pandas.DataFrame: Data with duplicate rows removed.
        """
        if self.data is None:
            print("No data loaded. Please load data before removing duplicates.")
            return None

        try:
            cleaned_data = self.data.drop_duplicates(subset=subset, keep=keep)
            return cleaned_data

        except Exception as e:
            print(f"An error occurred while removing duplicates: {str(e)}")
            return None

    def perform_groupby(self, by, agg_func):
        """
        Perform groupby operation on the loaded data.

        Args:
            by (str or list): Column(s) to group by.
            agg_func (str or dict): Aggregation function(s) to apply.

        Returns:
            pandas.DataFrame: Grouped data with applied aggregation functions.
        """
        if self.data is None:
            print("No data loaded. Please load data before performing groupby.")
            return None

        try:
            grouped_data = self.data.groupby(by=by).agg(agg_func)
            return grouped_data

        except Exception as e:
            print(f"An error occurred during groupby operation: {str(e)}")
            return None

    def rename_columns(self, column_map):
        """
        Rename columns of the loaded data.

        Args:
            column_map (dict): Dictionary mapping current column names to new names.

        Returns:
            bool: True if columns are successfully renamed, False otherwise.
        """
        if self.data is None:
            print("No data loaded. Please load data before renaming columns.")
            return False

        try:
            self.data.rename(columns=column_map, inplace=True)
            return True

        except Exception as e:
            print(f"An error occurred while renaming columns: {str(e)}")
            return False

    def calculate_average_sentiment(self, sentiment_column):
        """
        Calculate the average sentiment score for the loaded data.

        Args:
            sentiment_column (str): Name of the column containing the sentiment scores.

        Returns:
            float: Average sentiment score.
        """
        if self.data is None:
            print("No data loaded. Please load data before calculating average sentiment.")
            return None

        if sentiment_column not in self.data.columns:
            print(f"Failed to calculate average sentiment. Column '{sentiment_column}' does not exist in the data.")
            return None

        try:
            average_sentiment = np.mean(self.data[sentiment_column])
            return average_sentiment

        except Exception as e:
            print(f"An error occurred during average sentiment calculation: {str(e)}")
            return None

    def detect_outliers(self, sentiment_column, threshold=2):
        """
        Detect outliers based on the sentiment scores.

        Args:
            sentiment_column (str): Name of the column containing the sentiment scores.
            threshold (int or float): Threshold value to define outliers (default: 2).

        Returns:
            pandas.DataFrame: Data containing only the rows with outlier sentiment scores.
        """
        if self.data is None:
            print("No data loaded. Please load data before detecting outliers.")
            return None

        if sentiment_column not in self.data.columns:
            print(f"Failed to detect outliers. Column '{sentiment_column}' does not exist in the data.")
            return None

        try:
            outliers = self.data[self.data[sentiment_column] > threshold]
            return outliers

        except Exception as e:
            print(f"An error occurred while detecting outliers: {str(e)}")
            return None

    def calculate_correlation(self, column1, column2):
        """
        Calculate the correlation between two columns of the loaded data.

        Args:
            column1 (str): Name of the first column.
            column2 (str): Name of the second column.

        Returns:
            float: Correlation coefficient between the two columns.
        """
        if self.data is None:
            print("No data loaded. Please load data before calculating correlation.")
            return None

        if column1 not in self.data.columns or column2 not in self.data.columns:
            print("Failed to calculate correlation. One or more columns do not exist in the data.")
            return None

        try:
            correlation = self.data[column1].corr(self.data[column2])
            return correlation

        except Exception as e:
            print(f"An error occurred during correlation calculation: {str(e)}")
            return None


# Example usage
analyzer = SentimentAnalyzer()

# Load data from a CSV file
if analyzer.load_data("data.csv"):
    # Perform sentiment analysis on the 'text' column and store the results in the 'sentiment' column
    if analyzer.analyze_sentiment("text", "sentiment"):
        data = analyzer.get_data()
        print(data.head())
