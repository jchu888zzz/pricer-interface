from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QStackedWidget, QLabel, QProgressBar
)
from PySide6.QtCore import Qt, QTimer,QSize
from PySide6.QtGui import QMovie
import os

class Ui_LoadingPage(QWidget):
    def __init__(self):
        super().__init__()
        self.setup_ui()
    
    def setup_ui(self):
        self.setObjectName("LoadingPage")
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)

        self.loading_label = QLabel("Solving",self)
        self.loading_label.setAlignment(Qt.AlignCenter)
        self.loading_label.setObjectName("loadingLabel")
        layout.addWidget(self.loading_label)
        # Spinner (GIF or fallback)
        self.spinner_label = QLabel(self)
        self.spinner_label.setAlignment(Qt.AlignCenter)
        spinner_gif_path = os.path.join(os.path.dirname(__file__),"gifs" ,"spinner.gif")
        self._spinner_movie = None
        if os.path.exists(spinner_gif_path):
            self._spinner_movie = QMovie(spinner_gif_path)
            self.spinner_label.setMovie(self._spinner_movie)
            #self._spinner_movie.setScaledSize(QSize(450,450))
            self._spinner_movie.start()
        else:
            # fallback: indeterminate QProgressBar
            self.spinner_bar = QProgressBar(self)
            self.spinner_bar.setRange(0, 0)
            self.spinner_bar.setTextVisible(False)
            self.spinner_bar.setFixedWidth(120)
            layout.addWidget(self.spinner_bar)
        layout.addWidget(self.spinner_label)

        # Animated dots for loading (kept for text)
        self._dots = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._animate_loading)
        self._timer.start(400)
    
    def _animate_loading(self):
        self._dots = (self._dots + 1) % 4
        base=self.loading_label.text().split(".")[0]
        self.loading_label.setText(base + "." * self._dots)
