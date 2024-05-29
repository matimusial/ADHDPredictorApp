import os
import tempfile
import mysql.connector
from mysql.connector import Error
from tensorflow.keras.models import load_model

class DBConnector:
    def __init__(self):
        try:
            host = '192.168.205.218'
            database = 'mydatabase'
            user = 'user'
            password = 'user'
            self.connection = mysql.connector.connect(
                host=host,
                database=database,
                user=user,
                password=password
            )
            if self.connection.is_connected():
                self.cursor = self.connection.cursor()
                print("Połączenie do bazy danych zostało nawiązane.")
        except Error as e:
            print(f"Błąd połączenia: {e}")
            print("Pamiętaj o włączeniu ZUT VPN.")
            self.connection = None

    def __del__(self):
        if self.connection and self.connection.is_connected():
            self.connection.close()
            print("Połączenie do bazy danych zostało zamknięte.")

    def validate_and_convert_input_models(self, file_data, channels, input_shape, type_value, fs, plane, description):
        """
        Validates input for models table
        """
        if not file_data.endswith('.keras'):
            print("Błąd: 'file' musi mieć rozszerzenie '.keras'.")
            return None

        if not isinstance(channels, int):
            print("Błąd: 'channels' musi być liczbą całkowitą.")
            return None

        if not (isinstance(input_shape, tuple) and all(isinstance(i, int) for i in input_shape)):
            print("Błąd: 'input_shape' musi być tuplą z liczbami całkowitymi.")
            return None

        type_value = type_value.lower()
        if type_value not in ['cnn_mri', 'cnn_eeg', 'gan']:
            print("Błąd: 'type' musi być jedną z wartości 'cnn_mri', 'cnn_eeg', 'gan'.")
            return None

        if not isinstance(fs, (float, int)):
            print("Błąd: 'fs' musi być liczbą zmiennoprzecinkową lub całkowitą.")
            return None

        if plane not in ['A', 'S', 'C', '']:
            print("Błąd: 'plane' musi być jedną z wartości 'A', 'S', 'C', "" .")
            return None

        if len(description) > 254:
            print("Błąd: 'description' nie może być dłuższy niż 254 znaki.")
            return None

        return file_data, channels, str(input_shape), type_value, float(fs), plane, description

    def insert_data_into_models(self, name="none", file_path="none", channels=0, input_shape=(0, 0, 0),
                                type_value="none", fs=0.0, plane="none", description="none"):
        """
        Inserts data into the 'models' table.

        Args:
            name (str): Lowercase name of the model.
            file_path (str): Path to the model file (.keras).
            channels (int): The channels.
            input_shape (tuple): The input shape of the model.
            type_value (str): The type of the model - only ['cnn_mri', 'cnn_eeg', 'gan']
            fs (float): The sampling frequency.
            plane (str): The plane of the model - only ['A', 'S', 'C', ""]
            description (str)

        Returns:
            None
        """
        validated_data = self.validate_and_convert_input_models(file_path, channels, input_shape, type_value, fs, plane,
                                                                description)
        if not validated_data:
            print("Nieprawidłowe dane wejściowe.")
            return

        file_path, channels, input_shape_str, type_value, fs, plane, description = validated_data

        try:
            with open(file_path, 'rb') as file:
                model_data = file.read()
        except Exception as e:
            print(f"Błąd podczas wczytywania modelu: {e}")
            return

        if self.connection and self.connection.is_connected():
            try:
                query = """
                INSERT INTO models (name, file, channels, input_shape, type, fs, plane, description) 
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """
                self.cursor.execute(query, (
                    name.lower(), model_data, channels, input_shape_str, type_value, fs, plane, description))
                self.connection.commit()
                print("Dane zostały pomyślnie wstawione do tabeli models.")
            except Error as e:
                print(f"Błąd podczas wstawiania danych: {e}")
        else:
            print("Brak połączenia z bazą danych.")

    def select_data(self, table_name=""):
        """
        Selects and returns all data from the specified table.

        Args:
            table_name (str): The name of the table to select data from.

        Returns:
            list: A list of tuples containing the data from the table, or None if an error occurs.
        """
        if self.connection and self.connection.is_connected():
            try:
                query = f"SELECT * FROM {str(table_name)}"
                self.cursor.execute(query)
                results = self.cursor.fetchall()
                return results
            except Error as e:
                print(f"Błąd podczas pobierania danych: {e}")
                return None
        else:
            print("Brak połączenia z bazą danych.")
            return None

    def select_model_name(self, condition=""):
        """
        Selects and returns data from the 'models' table based on a condition.

        Args:
            condition (str): The condition to filter the models.
            example: "WHERE cond1 = 'val1' AND/OR cond2 = 'val'" check insert_data_for_models for table structure.

        Returns:
            list: A list of tuples containing the data from the table, or None if an error occurs.
        """
        if self.connection and self.connection.is_connected():
            try:
                query = f"SELECT name FROM models WHERE {condition}"
                self.cursor.execute(query)
                results = self.cursor.fetchall()
                return results

            except Error as e:
                print(f"Błąd podczas pobierania danych: {e}")
                return None
        else:
            print("Brak połączenia z bazą danych.")
            return None

    def select_model(self, model_name=""):
        """
        Selects and returns the Keras model for a specified model name.

        Args:
            model_name (str): The name of the model.

        Returns:
            model: The Keras model object, or None if an error occurs.
        """
        if self.connection and self.connection.is_connected():
            try:
                query = "SELECT file FROM models WHERE name=%s"
                self.cursor.execute(query, (model_name,))
                results = self.cursor.fetchall()

                if results:
                    file_data = results[0][0]  # Zakładamy, że pierwszy wynik to właściwy BLOB danych

                    # Tworzenie tymczasowego katalogu
                    with tempfile.TemporaryDirectory() as tmpdirname:
                        file_path = os.path.join(tmpdirname, f"{model_name}.keras")

                        # Zapis danych BLOB do pliku .keras
                        with open(file_path, 'wb') as file:
                            file.write(file_data)

                        # Odczyt modelu Keras z zapisanego pliku
                        model = load_model(file_path)

                    print(f"Model {model_name} został pomyślnie załadowany.")
                    return model
                else:
                    print("Nie znaleziono modelu o podanej nazwie.")
                    return None
            except Exception as e:
                print(f"Błąd podczas pobierania danych: {e}")
                return None
        else:
            print("Brak połączenia z bazą danych.")
            return None
