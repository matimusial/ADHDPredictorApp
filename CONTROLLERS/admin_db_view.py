from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QTableWidgetItem, QMessageBox, QHeaderView
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
        self.set_range_and_update_label()

    def set_range_and_update_label(self):
        """
        Sets the range for the spinBox widget and updates the table widget with model data
        from the database.
        """
        try:
            self.db.establish_connection()
        except ConnectionError:
            self.show_alert("Cannot establish database connection, remember to enable ZUT VPN.")
            return
        self.id_table = []
        data, column_names = self.db.select_data_and_columns("models")

        min_id = data[0][0]
        max_id = data[-1][0]
        for record in data:
            self.id_table.append(record[0])
        self.ui.spinBox.setRange(min_id, max_id)

        self.ui.tableWidget.setRowCount(len(data))
        self.ui.tableWidget.setColumnCount(len(column_names))
        self.ui.tableWidget.setHorizontalHeaderLabels(column_names)
        self.ui.tableWidget.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)

        for row_num, row_data in enumerate(data):
            for col_num, col_data in enumerate(row_data):
                item = QTableWidgetItem(str(col_data))
                item.setTextAlignment(Qt.AlignCenter)
                item.setFlags(Qt.ItemIsEnabled)
                self.ui.tableWidget.setItem(row_num, col_num, item)

        self.ui.tableWidget.resizeColumnsToContents()
        self.ui.tableWidget.resizeRowsToContents()

        header = self.ui.tableWidget.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Fixed)
        header.setSectionResizeMode(column_names.index('description'), QHeaderView.Stretch)

        vertical_header = self.ui.tableWidget.verticalHeader()
        vertical_header.setSectionResizeMode(QHeaderView.Stretch)

        self.ui.tableWidget.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)

        self.ui.tableWidget.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

    def delete_row(self, id_table):
        """
        Deletes a row from the models table in the database and updates the view.

        Args:
            id_table (int): The ID of the model to be deleted.
        """
        try:
            self.db.establish_connection()
        except ConnectionError:
            self.show_alert("Cannot establish database connection, remember to enable ZUT VPN.")
            return
        self.db.delete_data_from_models_table(id_table)
        self.set_range_and_update_label()

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

    def show_alert(self, msg):
        """
        Displays a warning message.

        Args:
            msg (str): The content of the warning message.
        """
        alert = QMessageBox()
        alert.setWindowTitle("Warning")
        alert.setText(msg)
        alert.setIcon(QMessageBox.Warning)
        alert.setStandardButtons(QMessageBox.Ok)
        alert.exec_()
