from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QTableWidgetItem, QMessageBox
from PyQt5.QtCore import Qt

from CONTROLLERS.DBConnector import DBConnector


class AdminDbView:
    """
    This class represents the administrative database view for managing model data.
    """

    def __init__(self, ui):
        """
        Initializes the AdminDbView class.

        Args:
            ui (QMainWindow): The main window UI object.
        """
        self.ui = ui
        self.db = DBConnector()
        self.id_table = None
        self.set_range()
        self.update_label()

    def set_range(self):
        """
        Sets the range for the spinBox widget based on the model IDs present in the database.
        """
        self.id_table = []
        self.db.establish_connection()
        data = self.db.select_data_and_columns("models")[0]
        min_id = data[0][0]
        max_id = data[len(data)-1][0]
        for i in range(len(data)):
            self.id_table.append(data[i][0])
        self.ui.spinBox.setRange(min_id, max_id)
        self.ui.spinBox.setValue(max_id)

    def update_label(self):
        """
        Updates the table widget with model data from the database.
        """
        self.db.establish_connection()
        data, column_names = self.db.select_data_and_columns("models")

        self.ui.tableWidget.setRowCount(len(data))
        self.ui.tableWidget.setColumnCount(len(column_names))
        self.ui.tableWidget.setHorizontalHeaderLabels(column_names)
        self.ui.tableWidget.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)

        for row_num, row_data in enumerate(data):
            for col_num, col_data in enumerate(row_data):
                item = QTableWidgetItem(str(col_data))
                item.setFlags(Qt.ItemIsEnabled)
                self.ui.tableWidget.setItem(row_num, col_num, item)

        self.ui.tableWidget.resizeColumnsToContents()

    def delete_row(self, id):
        """
        Deletes a row from the models table in the database and updates the view.

        Args:
            id (int): The ID of the model to be deleted.
        """
        self.db.establish_connection()
        self.db.delete_data_from_models_table(id)
        self.set_range()
        self.update_label()

    def show_dialog(self):
        """
        Shows a confirmation dialog for deleting a model and performs the deletion if confirmed.
        """
        model_id = self.ui.spinBox.value()
        alert = QMessageBox()
        alert.setWindowTitle("Warning")
        alert.setIcon(QMessageBox.Warning)
        if model_id not in self.id_table:
            alert.setText(f"ID: {model_id} does not exist!")
            alert.setStandardButtons(QMessageBox.Ok)
            alert.exec_()
        else:
            alert.setText(f"Are you sure you want to delete model {model_id}?")
            alert.setStandardButtons(QMessageBox.Yes | QMessageBox.No)

            response = alert.exec_()
            if response == QMessageBox.Yes:
                self.delete_row(model_id)
            else:
                return
