import mysql.connector
from mysql.connector import Error


class DBConnector:
    def __init__(self):
        self.connection = None

    def __del__(self):
        if self.connection and self.connection.is_connected():
            self.connection.close()
            print("Database connection closed.")

    def establish_connection(self):
        if self.connection and self.connection.is_connected():
            return
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
                print("Database connection established.")
            else:
                raise Error("Failed to establish database connection.")
        except Error as e:
            print(f"Connection error: {e}")
            print("Remember to enable ZUT VPN.")
            raise Error("Failed to establish database connection, remember to enable ZUT VPN.")

    def validate_and_convert_input_models(self, name, file_data, channels, input_shape, type_value, fs, plane,
                                          description):
        """
        Validates input for models table
        """
        if not all(char.isdigit() or char == '.' for char in name):
            print("Error: 'name' must consist only of digits (0-9) and/or dots ('.').")
            return

        if not file_data.endswith('.keras'):
            print("Error: 'model path' must have '.keras' extension.")
            return

        if channels is not None and not isinstance(channels, int):
            print("Error: 'channels' must be an integer or None.")
            return

        if not (isinstance(input_shape, tuple) and all(isinstance(i, int) for i in input_shape)):
            print("Error: 'input_shape' must be a tuple of integers.")
            return

        type_value = type_value.lower()
        if type_value not in ['cnn_mri', 'cnn_eeg', 'gan_adhd', 'gan_control']:
            print("Error: 'type' must be one of 'cnn_mri', 'cnn_eeg', 'gan_adhd', 'gan_control'.")
            return

        if fs is not None:
            if not isinstance(fs, (float, int)):
                print("Error: 'fs' must be a float, integer, or None.")
                return
            if fs == 0:
                print("Error: 'fs' cannot be 0.")
                return

        if plane is not None and plane not in ['A', 'S', 'C']:
            print("Error: 'plane' must be one of 'A', 'S', 'C' or None.")
            return

        if len(description) > 254:
            print("Error: 'description' cannot be longer than 254 characters.")
            return

        channels = channels if channels is not None else None
        fs = fs if fs is not None else None
        plane = plane if plane is not None else None

        return name, file_data, channels, str(input_shape), type_value, float(
            fs) if fs is not None else None, plane, description

    def insert_data_into_models_table(self, name="none", file_path="none", channels=None, input_shape=(0, 0, 0),
                                      type_value="none", fs=None, plane=None, description="none"):
        """
        Inserts data into the 'models' and 'files' tables.

        Args:
            name (str): Accuracy of the model.
            file_path (str): Path to the model (.keras).
            channels (int): The channels or None
            input_shape (tuple): The input shape of the model.
            type_value (str): The type of the model - only ['cnn_mri', 'cnn_eeg', 'gan_adhd', 'gan_control']
            fs (float): The sampling frequency or None
            plane (str): The plane of the model - only ['A', 'S', 'C', None] (axial, sagittal, coronal)
            description (str)

        Returns:
            None
        """
        validated_data = self.validate_and_convert_input_models(name, file_path, channels, input_shape, type_value, fs, plane, description)
        if not validated_data:
            print("Invalid input data.")
            return

        name, file_path, channels, input_shape_str, type_value, fs, plane, description = validated_data

        try:
            with open(file_path, 'rb') as file:
                model_data = file.read()
        except Exception as e:
            print(f"Error loading model: {e}")
            return

        if self.connection and self.connection.is_connected():
            try:
                query_models = """
                INSERT INTO models (name, channels, input_shape, type, fs, plane, description) 
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """
                self.cursor.execute(query_models, (
                    name, channels, input_shape_str, type_value, fs, plane, description))
                self.connection.commit()

                model_id = self.cursor.lastrowid

                query_files = """
                INSERT INTO files (model_id, file) 
                VALUES (%s, %s)
                """
                self.cursor.execute(query_files, (model_id, model_data))
                self.connection.commit()

                print(f"Model {type_value} {name} successfully inserted into 'models' and 'files' tables.")
            except Error as e:
                print(f"Error inserting data: {e}")
        else:
            print("No database connection, use establish_connection function.")

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
                print(f"Error retrieving data: {e}")
                return None
        else:
            print("No database connection, use establish_connection function.")
            return None

    def select_model_info(self, condition=""):
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
                query = f"SELECT name, input_shape, fs, channels, plane, description FROM models WHERE {condition} ORDER BY name DESC"
                self.cursor.execute(query)
                results = self.cursor.fetchall()
                return results
            except Error as e:
                print(f"Error retrieving data: {e}")
                return None
        else:
            print("No database connection, use establish_connection function.")
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
                            print("Model successfully loaded by Keras.")
                        except Exception as e:
                            print(f"Error loading model by Keras: {e}")
                            return None

                        if os.path.exists("tmp.keras"):
                            os.remove("tmp.keras")

                        return loaded_model
                    else:
                        print("No file found for the specified model.")
                        return None
                else:
                    print("No model found with the specified name.")
                    return None
            except Error as e:
                print(f"Error retrieving data: {e}")
                return None
        else:
            print("No database connection, use establish_connection function.")
            return None

    def delete_data_from_models_table(self, model_id):
        """
        Deletes data from the 'models' and 'files' tables based on the provided model ID.

        Args:
            model_id (int): The ID of the model to be deleted.

        Returns:
            None
        """
        if self.connection and self.connection.is_connected():
            try:
                query_files = """DELETE FROM files WHERE model_id = %s"""
                self.cursor.execute(query_files, (model_id,))
                self.connection.commit()

                query_models = """DELETE FROM models WHERE id = %s"""
                self.cursor.execute(query_models, (model_id,))
                self.connection.commit()

                print(f"Model with ID {model_id} successfully deleted from 'models' and 'files' tables.")
            except Error as e:
                print(f"Error deleting data: {e}")
        else:
            print("No database connection, use establish_connection function.")


# db = DBConnector()
# db.establish_connection()
