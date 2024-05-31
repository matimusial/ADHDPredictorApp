import mysql.connector
from mysql.connector import Error


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
            print("Błąd: 'model path' musi mieć rozszerzenie '.keras'.")
            return

        if channels is not None and not isinstance(channels, int):
            print("Błąd: 'channels' musi być liczbą całkowitą lub None.")
            return

        if not (isinstance(input_shape, tuple) and all(isinstance(i, int) for i in input_shape)):
            print("Błąd: 'input_shape' musi być tuplą z liczbami całkowitymi.")
            return

        type_value = type_value.lower()
        if type_value not in ['cnn_mri', 'cnn_eeg']:
            print("Błąd: 'type' musi być jedną z wartości 'cnn_mri', 'cnn_eeg'.")
            return

        if fs is not None:
            if not isinstance(fs, (float, int)):
                print("Błąd: 'fs' musi być liczbą zmiennoprzecinkową, całkowitą lub None.")
                return
            if fs == 0:
                print("Błąd: 'fs' nie może być równe 0.")
                return

        if plane is not None and plane not in ['A', 'S', 'C']:
            print("Błąd: 'plane' musi być jedną z wartości 'A', 'S', 'C' lub None.")
            return

        if len(description) > 254:
            print("Błąd: 'description' nie może być dłuższy niż 254 znaki.")
            return

        channels = channels if channels is not None else None
        fs = fs if fs is not None else None
        plane = plane if plane is not None else None

        return file_data, channels, str(input_shape), type_value, float(
            fs) if fs is not None else None, plane, description

    def insert_data_into_models(self, name="none", file_path="none", channels=None, input_shape=(0, 0, 0),
                                type_value="none", fs=None, plane=None, description="none"):
        """
        Inserts data into the 'models' and 'files' tables.

        Args:
            name (str): Lowercase name of the model.
            file_path (str): Path to the model (.keras).
            channels (int): The channels or None
            input_shape (tuple): The input shape of the model.
            type_value (str): The type of the model - only ['cnn_mri', 'cnn_eeg']
            fs (float): The sampling frequency or None
            plane (str): The plane of the model - only ['A', 'S', 'C', None]
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
                query_models = """
                INSERT INTO models (name, channels, input_shape, type, fs, plane, description) 
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """
                self.cursor.execute(query_models, (
                    name.lower(), channels, input_shape_str, type_value, fs, plane, description))
                self.connection.commit()

                model_id = self.cursor.lastrowid

                query_files = """
                INSERT INTO files (model_id, files) 
                VALUES (%s, %s)
                """
                self.cursor.execute(query_files, (model_id, model_data))
                self.connection.commit()

                print(f"Model {type} {name} został pomyślnie wstawiony do tabel models i files.")
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
        Selects and returns the data for a specified model name.

        Args:
            model_name (str): The name of the model.

        Returns:
            keras_file
        """
        from tensorflow.keras.models import load_model
        if self.connection and self.connection.is_connected():
            try:
                model_id_query = "SELECT id FROM models WHERE name=%s"
                self.cursor.execute(model_id_query, (model_name,))
                model_id_result = self.cursor.fetchall()

                if model_id_result:
                    model_id = model_id_result[0][0]
                    file_query = "SELECT file FROM files WHERE model_id=%s"
                    self.cursor.execute(file_query, (model_id,))
                    file_result = self.cursor.fetchall()

                    if file_result:
                        model_file_data = file_result[0][0]

                        import os
                        with open("tmp.keras", 'wb') as file:
                            file.write(model_file_data)

                        try:
                            loaded_model = load_model("tmp.keras")
                            print("Model został prawidłowo załadowany przez Keras.")
                        except Exception as e:
                            print(f"Błąd podczas ładowania modelu przez Keras: {e}")
                            return None

                        if os.path.exists("tmp.keras"):
                            os.remove("tmp.keras")

                        return loaded_model
                    else:
                        print("Nie znaleziono pliku dla podanego modelu.")
                        return None
                else:
                    print("Nie znaleziono modelu o podanej nazwie.")
                    return None
            except Error as e:
                print(f"Błąd podczas pobierania danych: {e}")
                return None
        else:
            print("Brak połączenia z bazą danych.")
            return None


#db = DBConnector()

#db.insert_data_into_models("0.7466", "../MRI/CNN/MODELS/0.7466.keras", None, (120, 120, 1), "cnn_mri",None, "A", "model testowy mri 6")
