from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QStackedWidget, QPushButton,QDialog,QSizePolicy
)
from PySide6.QtCore import Qt
import os

from app.ui_pages.LoadingPage import Ui_LoadingPage
from app.ui_pages.ResultPage import Ui_ResultPageEquity,Ui_ResultPageRate

class ResultDialog(QDialog):
    """
    A window with a QStackedWidget:
      - Page 0: Modern loading page (animated dots)
      - Page 1: Table display (dict to table)
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Results")
        self.setObjectName("ResultDialog")
        self.setMinimumSize(1000, 500)
        self.layout = QVBoxLayout(self)
        self.stack = QStackedWidget(self)
        self.layout.addWidget(self.stack)

        # --- Loading Page ---
        self.loading_page = Ui_LoadingPage()
        self.stack.addWidget(self.loading_page)

        # --- Table Page ---
        self.result_page=Ui_ResultPageRate()
        self.stack.addWidget(self.result_page)

        self._apply_style_from_qss()

    def _apply_style_from_qss(self, path: str = None):
        """
        Load a .qss file and apply as the app stylesheet.
        Default path: src/app/styles/theme.qss (project-relative).        """
        if path is None:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            path = os.path.join(project_root,"app", "styles", "theme.qss")
        try:
            with open(path, "r", encoding="utf-8") as f:
                self.setStyleSheet(f.read())
        except Exception:
            pass

    def show_loading_page(self,parent: QWidget | None = None):
        self.stack.setCurrentIndex(0)
        self.loading_page._timer.start(400)
        if parent is not None:
            self.setParent(parent, Qt.Window)
            self._center_on_parent(parent)
        # Start spinner animation if present
        if self.loading_page._spinner_movie:
            self.loading_page._spinner_movie.start()
        self.setModal(False)
        self.show()

    def show_table_page(self, data:tuple[dict]):
        self.stack.setCurrentIndex(1)
        self.loading_page._timer.stop()
        # Stop spinner animation if present
        if self.loading_page._spinner_movie:
            self.loading_page._spinner_movie.stop()
        if data:
            self.result_page.retrieve_data(data[0],data[1])
            #self.result_page.set_data(data)
            self.adjustSize()

    def _center_on_parent(self, parent: QWidget):
        # center this dialog on the parent widget
        parent_rect = parent.frameGeometry()
        self_rect = self.frameGeometry()
        center = parent_rect.center()
        self_rect.moveCenter(center)
        self.move(self_rect.topLeft())

class ResultDialogEquity(QDialog):
    """
    A window with a QStackedWidget:
      - Page 0: Modern loading page (animated dots)
      - Page 1: Table display (dict to table)
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Results")
        self.setObjectName("ResultDialog")
        self.setMinimumSize(700, 400)
        self.layout = QVBoxLayout(self)
        self.stack = QStackedWidget(self)
        self.layout.addWidget(self.stack)

        # --- Loading Page ---
        self.loading_page = Ui_LoadingPage()
        self.stack.addWidget(self.loading_page)

        # --- Table Page ---
        self.result_page = Ui_ResultPageEquity()
        self.stack.addWidget(self.result_page)


        self._apply_style_from_qss()

    def _apply_style_from_qss(self, path: str = None):
        """
        Load a .qss file and apply as the app stylesheet.
        Default path: src/app/styles/theme.qss (project-relative).        """
        if path is None:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            path = os.path.join(project_root,"app", "styles", "theme.qss")
        try:
            with open(path, "r", encoding="utf-8") as f:
                self.setStyleSheet(f.read())
        except Exception:
            pass

    def show_loading_page(self,parent: QWidget | None = None):
        self.stack.setCurrentIndex(0)
        self.loading_page._timer.start(400)
        if parent is not None:
            self.setParent(parent, Qt.Window)
            self._center_on_parent(parent)
        # Start spinner animation if present
        if self.loading_page._spinner_movie:
            self.loading_page._spinner_movie.start()
        self.setModal(False)
        self.show()

    def show_table_page(self, data:dict |None):
        self.stack.setCurrentIndex(1)
        self.loading_page._timer.stop()
        # Stop spinner animation if present
        if self.loading_page._spinner_movie:
            self.loading_page._spinner_movie.stop()
        if data:
            self.result_page.set_data(data)
            self.adjustSize()

    def _center_on_parent(self, parent: QWidget):
        # center this dialog on the parent widget
        parent_rect = parent.frameGeometry()
        self_rect = self.frameGeometry()
        center = parent_rect.center()
        self_rect.moveCenter(center)
        self.move(self_rect.topLeft())