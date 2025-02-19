from PyQt6.QtWidgets import QMainWindow, QLabel, QVBoxLayout, QWidget, QProgressBar
from PyQt6.QtCore import Qt, QRect, QSize
from PyQt6.QtGui import QMovie
from GUI.gridsearchworker2 import GridSearchWorker

class GridSearchGUI(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Grid Search Progress")
        self.setGeometry(100, 100, 400, 200)

        self.layout = QVBoxLayout()

        # Add GIF
        self.gif_label = QLabel(self)
        self.gif_label.setGeometry(QRect(50, 50, 200, 200))
        self.gif_label.setMinimumSize(QSize(125, 100))
        self.gif_label.setMaximumSize(QSize(500, 500))
        self.movie = QMovie('/Users/enricadillon/Python/N2predicter/GUI/cloud.gif')
        self.movie.setScaledSize(QSize(100, 100))  # Adjust size here
        self.gif_label.setMovie(self.movie)
        self.movie.start()
        self.gif_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.gif_label)

        # Add progress bar
        self.progress_label = QLabel("Progress: 0%", self)
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.progress_label)
        self.progress_bar = QProgressBar(self)
        self.layout.addWidget(self.progress_bar)

        # Add best accuracy label
        self.best_accuracy_label = QLabel("Best Accuracy: N/A", self)
        self.best_accuracy_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.best_accuracy_label)

        # Add best parameters label
        self.best_params_label = QLabel("Best Parameters: N/A", self)
        self.best_params_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.best_params_label)

        # Set layout
        container = QWidget()
        container.setLayout(self.layout)
        self.setCentralWidget(container)

    def start_grid_search(self, training_data, test_data, param_grid, epochs, mini_batch_size):
        self.worker = GridSearchWorker(training_data, test_data, param_grid, epochs, mini_batch_size)
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.best_accuracy_signal.connect(self.update_best_accuracy)
        self.worker.best_params_signal.connect(self.update_best_params)
        self.worker.start()

    def update_progress(self, progress, total):
        self.progress_bar.setValue(int((progress / total) * 100))
        self.progress_label.setText(f"Progress: {int((progress / total) * 100)}%")

    def update_best_accuracy(self, best_accuracy):
        self.best_accuracy_label.setText(f"best accuracy: {best_accuracy: .4f}")

    def update_best_params(self, best_params):
        formatted_params = "\n".join([f"{key}: {value}" for key, value in best_params.items()])
        self.best_params_label.setText(f"best parameters:\n{formatted_params}")