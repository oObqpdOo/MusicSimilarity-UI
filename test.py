from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog
from PyQt5 import uic
import sys

form_class = uic.loadUiType("ms.ui")[0]  # Load the UI

class MyWindowClass(QMainWindow, form_class):
    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)
        self.setupUi(self)


    def extrbutton_clicked(self):
        options = QFileDialog.Options()
        fileName = QFileDialog.getExistingDirectory(self,"Select folder containing music", "", options=options)
        if fileName:
            print(fileName)

    def loadbutton_clicked(self):
        options = QFileDialog.Options()
        fileName = QFileDialog.getExistingDirectory(self,"Select folder containing feature files", "","All Files (*)", options=options)
        if fileName:
            print(fileName)

    def selectbutton_clicked(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self,"Select a song", "","All Files (*)", options=options)
        if fileName:
            print(fileName)

app = QApplication(sys.argv)
myWindow = MyWindowClass(None)
myWindow.show()
app.exec_()
