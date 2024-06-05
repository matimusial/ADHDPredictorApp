from DBConnector import DBConnector

class generateNew:
    def __init__(self, ui):
        self.ui = ui
        self.db = DBConnector


    def showInfo(self, data):
        self.db.establish_connection()
        self.db.select_model_name()

        pass

    def generate(self):
        print("generate")