import yaml
import psycopg2
import pymysql
import pymongo
from pymongo.errors import PyMongoError
import pyodbc
import cx_Oracle
import ibm_db


def load_credentials():
    """Load the database credentials from credentials.yaml file."""
    with open('credentials.yaml', 'r') as file:
        credentials = yaml.safe_load(file)
    return credentials


class PostgresqlConnection:
    def __enter__(self):
        credentials = load_credentials()
        self.connection = psycopg2.connect(
            host=credentials['postgresql']['host'],
            port=credentials['postgresql']['port'],
            database=credentials['postgresql']['database'],
            user=credentials['postgresql']['user'],
            password=credentials['postgresql']['password']
        )
        return self.connection

    def __exit__(self, exc_type, exc_value, traceback):
        self.connection.close()

    def execute_query(self, query):
        """Execute a SQL query on the PostgreSQL database."""
        try:
            cursor = self.connection.cursor()
            cursor.execute(query)
            result = cursor.fetchall()
            cursor.close()
            return result
        except (psycopg2.Error, Exception) as e:
            # Handle the exception here or raise it to the caller
            print(f"An error occurred while executing the query: {e}")
            return None

    def insert_data(self, table, data):
        """Insert data into the specified table in the PostgreSQL database."""
        try:
            cursor = self.connection.cursor()
            columns = ', '.join(data.keys())
            values = ', '.join(['%s'] * len(data))
            query = f"INSERT INTO {table} ({columns}) VALUES ({values})"
            cursor.execute(query, tuple(data.values()))
            self.connection.commit()
            cursor.close()
        except (psycopg2.Error, Exception) as e:
            # Handle the exception here or raise it to the caller
            print(f"An error occurred while inserting data: {e}")
            return None

    def update_data(self, table, condition, data):
        """Update data in the specified table in the PostgreSQL database based on the given condition."""
        try:
            cursor = self.connection.cursor()
            set_values = ', '.join([f"{column} = %s" for column in data.keys()])
            query = f"UPDATE {table} SET {set_values} WHERE {condition}"
            cursor.execute(query, tuple(data.values()))
            self.connection.commit()
            cursor.close()
        except (psycopg2.Error, Exception) as e:
            # Handle the exception here or raise it to the caller
            print(f"An error occurred while updating data: {e}")

    def delete_data(self, table, condition):
        """Delete data from the specified table in the PostgreSQL database based on the given condition."""
        try:
            cursor = self.connection.cursor()
            query = f"DELETE FROM {table} WHERE {condition}"
            cursor.execute(query)
            self.connection.commit()
            cursor.close()
        except (psycopg2.Error, Exception) as e:
            # Handle the exception here or raise it to the caller
            print(f"An error occurred while deleting data: {e}")


class MysqlConnection:
    def __enter__(self):
        credentials = load_credentials()
        self.connection = pymysql.connect(
            host=credentials['mysql']['host'],
            port=credentials['mysql']['port'],
            database=credentials['mysql']['database'],
            user=credentials['mysql']['user'],
            password=credentials['mysql']['password']
        )
        return self.connection

    def __exit__(self, exc_type, exc_value, traceback):
        self.connection.close()

    def execute_query(self, query):
        """Execute a SQL query on the MySQL database."""
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query)
                result = cursor.fetchall()
            return result
        except Exception as e:
            print(f"Error executing query: {e}")
            return None

    def insert_data(self, table, data):
        """Insert data into the specified table in MySQL."""
        try:
            with self.connection.cursor() as cursor:
                columns = ', '.join(data.keys())
                values = ', '.join(['%s'] * len(data))
                query = f"INSERT INTO {table} ({columns}) VALUES ({values})"
                cursor.execute(query, tuple(data.values()))
                self.connection.commit()
                return cursor.lastrowid
        except Exception as e:
            print(f"Error inserting data: {e}")
            return None

    def update_data(self, table, condition, data):
        """Update data in the specified table in MySQL based on the given condition."""
        try:
            with self.connection.cursor() as cursor:
                set_values = ', '.join([f"{column} = %s" for column in data.keys()])
                query = f"UPDATE {table} SET {set_values} WHERE {condition}"
                cursor.execute(query, tuple(data.values()))
                self.connection.commit()
        except Exception as e:
            print(f"Error updating data: {e}")

    def delete_data(self, table, condition):
        """Delete data from the specified table in MySQL based on the given condition."""
        try:
            with self.connection.cursor() as cursor:
                query = f"DELETE FROM {table} WHERE {condition}"
                cursor.execute(query)
                self.connection.commit()
        except Exception as e:
            print(f"Error deleting data: {e}")


class MongodbConnection:
    def __enter__(self):
        credentials = load_credentials()
        client = pymongo.MongoClient(
            host=credentials['mongodb']['host'],
            port=credentials['mongodb']['port'],
            username=credentials['mongodb']['user'],
            password=credentials['mongodb']['password']
        )
        self.database = client[credentials['mongodb']['database']]
        return self.database

    def __exit__(self, exc_type, exc_value, traceback):
        self.database.client.close()

    def execute_query(self, query):
        """Execute a SQL-like query using MongoDB's aggregation framework."""
        try:
            result = self.database.command('aggregate', query)
            return result['result']
        except PyMongoError as e:
            print(f"Error executing SQL query in MongoDB: {e}")
            return None

    def insert_data(self, collection_name, data):
        """Insert data into the specified collection in MongoDB."""
        try:
            collection = self.database[collection_name]
            result = collection.insert_one(data)
            return result.inserted_id
        except PyMongoError as e:
            print(f"Error inserting data into MongoDB: {e}")
            return None

    def update_data(self, collection_name, condition, values):
        """Update data in the specified collection in MongoDB based on the given condition."""
        try:
            collection = self.database[collection_name]
            collection.update_many(condition, {"$set": values})
        except PyMongoError as e:
            print(f"Error updating data in MongoDB: {e}")

    def delete_data(self, collection_name, condition):
        """Delete data from the specified collection in MongoDB based on the given condition."""
        try:
            collection = self.database[collection_name]
            collection.delete_many(condition)
        except PyMongoError as e:
            print(f"Error deleting data from MongoDB: {e}")


class SqlServerConnection:
    def __enter__(self):
        credentials = load_credentials()
        connection_string = f"DRIVER={{ODBC Driver 17 for SQL Server}};" \
                            f"SERVER={credentials['sql_server']['server']};" \
                            f"DATABASE={credentials['sql_server']['database']};" \
                            f"UID={credentials['sql_server']['user']};" \
                            f"PWD={credentials['sql_server']['password']}"
        self.connection = pyodbc.connect(connection_string)
        return self.connection

    def __exit__(self, exc_type, exc_value, traceback):
        self.connection.close()

    def execute_query(self, query):
        """Execute a SQL query and return the result."""
        try:
            cursor = self.connection.cursor()
            cursor.execute(query)
            result = cursor.fetchall()
            cursor.close()
            return result
        except Exception as e:
            print(f"Error executing query: {str(e)}")
            return None

    def insert_data(self, table, data):
        """Insert data into the specified table."""
        try:
            cursor = self.connection.cursor()
            columns = ', '.join(data.keys())
            placeholders = ', '.join('?' * len(data))
            values = list(data.values())
            query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
            cursor.execute(query, values)
            self.connection.commit()
            cursor.close()
        except Exception as e:
            print(f"Error inserting data: {str(e)}")
            return None

    def update_data(self, table, condition, data):
        """Update data in the specified table based on the given condition."""
        try:
            cursor = self.connection.cursor()
            set_clause = ', '.join([f"{column} = ?" for column in data.keys()])
            values = list(data.values())
            query = f"UPDATE {table} SET {set_clause} WHERE {condition}"
            cursor.execute(query, values)
            self.connection.commit()
            cursor.close()
        except Exception as e:
            print(f"Error updating data: {str(e)}")

    def delete_data(self, table, condition):
        """Delete data from the specified table based on the given condition."""
        try:
            cursor = self.connection.cursor()
            query = f"DELETE FROM {table} WHERE {condition}"
            cursor.execute(query)
            self.connection.commit()
            cursor.close()
        except Exception as e:
            print(f"Error deleting data: {str(e)}")


class OracleConnection:
    def __enter__(self):
        credentials = load_credentials()
        dsn = cx_Oracle.makedsn(
            credentials['oracle']['host'],
            credentials['oracle']['port'],
            service_name=credentials['oracle']['service_name']
        )
        self.connection = cx_Oracle.connect(
            user=credentials['oracle']['user'],
            password=credentials['oracle']['password'],
            dsn=dsn
        )
        return self.connection

    def __exit__(self, exc_type, exc_value, traceback):
        self.connection.close()

    def execute_query(self, query):
        """Execute a SQL query on the Oracle database."""
        try:
            cursor = self.connection.cursor()
            cursor.execute(query)
            result = cursor.fetchall()
            cursor.close()
            return result
        except cx_Oracle.Error as error:
            print(f"Error executing SQL query: {error}")
            return None

    def insert_data(self, table, data):
        """Insert data into the specified table in the Oracle database."""
        try:
            cursor = self.connection.cursor()
            columns = ", ".join(data.keys())
            placeholders = ":" + ", :".join(data.keys())
            query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
            cursor.execute(query, data)
            self.connection.commit()
            cursor.close()
        except cx_Oracle.Error as error:
            print(f"Error inserting data: {error}")
            return None

    def update_data(self, table, condition, values):
        """Update data in the specified table in the Oracle database based on the given condition."""
        try:
            cursor = self.connection.cursor()
            set_clause = ", ".join([f"{column} = :{column}" for column in values.keys()])
            query = f"UPDATE {table} SET {set_clause} WHERE {condition}"
            cursor.execute(query, values)
            self.connection.commit()
            cursor.close()
        except cx_Oracle.Error as error:
            print(f"Error updating data: {error}")

    def delete_data(self, table, condition):
        """Delete data from the specified table in the Oracle database based on the given condition."""
        try:
            cursor = self.connection.cursor()
            query = f"DELETE FROM {table} WHERE {condition}"
            cursor.execute(query)
            self.connection.commit()
            cursor.close()
        except cx_Oracle.Error as error:
            print(f"Error deleting data: {error}")


class Db2Connection:
    def __enter__(self):
        credentials = load_credentials()
        connection_string = f"DATABASE={credentials['db2']['database']};" \
                            f"HOSTNAME={credentials['db2']['hostname']};" \
                            f"PORT={credentials['db2']['port']};" \
                            f"PROTOCOL=TCPIP;" \
                            f"UID={credentials['db2']['user']};" \
                            f"PWD={credentials['db2']['password']}"
        self.connection = ibm_db.connect(connection_string, "", "")
        return self.connection

    def __exit__(self, exc_type, exc_value, traceback):
        ibm_db.close(self.connection)

    def execute_query(self, query):
        """Execute a SQL query on the DB2 database."""
        try:
            statement = ibm_db.prepare(self.connection, query)
            ibm_db.execute(statement)
            result = ibm_db.fetch_tuple(statement)
            return result
        except Exception as e:
            print(f"Error executing SQL query: {e}")
            return None

    def insert_data(self, table, data):
        """Insert data into the specified table in the DB2 database."""
        try:
            columns = ', '.join(data.keys())
            placeholders = ', '.join(['?' for _ in range(len(data))])
            values = tuple(data.values())
            query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
            statement = ibm_db.prepare(self.connection, query)
            ibm_db.execute(statement, values)
            return True
        except Exception as e:
            print(f"Error inserting data: {e}")
            return False

    def update_data(self, table, condition, data):
        """Update data in the specified table in the DB2 database based on the given condition."""
        try:
            set_values = ', '.join([f"{column} = ?" for column in data.keys()])
            values = tuple(data.values())
            query = f"UPDATE {table} SET {set_values} WHERE {condition}"
            statement = ibm_db.prepare(self.connection, query)
            ibm_db.execute(statement, values)
            return True
        except Exception as e:
            print(f"Error updating data: {e}")
            return False

    def delete_data(self, table, condition):
        """Delete data from the specified table in the DB2 database based on the given condition."""
        try:
            query = f"DELETE FROM {table} WHERE {condition}"
            statement = ibm_db.prepare(self.connection, query)
            ibm_db.execute(statement)
            return True
        except Exception as e:
            print(f"Error deleting data: {e}")
            return False


# Example usage
if __name__ == '__main__':
    with PostgresqlConnection() as postgresql_connection:
        # Use the PostgreSQL connection here
        pass

    with MysqlConnection() as mysql_connection:
        # Use the MySQL connection here
        pass

    with MongodbConnection() as mongodb_connection:
        # Use the MongoDB connection here
        pass

    with SqlServerConnection() as sql_server_connection:
        # Use the SQL Server connection here
        pass

    with OracleConnection() as oracle_connection:
        # Use the Oracle connection here
        pass

    with Db2Connection() as db2_connection:
        # Use the DB2 connection here
        pass
