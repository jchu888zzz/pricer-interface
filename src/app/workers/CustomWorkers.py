from PySide6.QtCore import QObject,Signal,Slot,QThread
from queue import Queue

import QuantLib as ql
from typing import Callable
import traceback
import Pricing.Rates.GetResults as RateGetResults

class MktDataWorker(QObject):
    """Worker to load market data from Excel files."""
    
    # Signals
    completed = Signal(dict)  # result
    failed = Signal(Exception)  # error
    
    def __init__(self, path_folder: str, date: ql.Date):
        super().__init__()
        self.path_folder = path_folder
        self.date =date
    def run(self):
        """Load market data with fallback to previous business day."""
        try:
            data = self._load_with_fallback(self.date)
            self.completed.emit(data)
        except Exception as e:
            self.failed.emit(e)
            traceback.print_exc()
    
    def _load_with_fallback(self, date: ql.Date, max_retries: int = 5) -> dict:
        """Try to load data for date, fallback to previous business days."""
        for _ in range(max_retries):
            try:
                return RateGetResults.retrieve_data(self.path_folder, date)
            except (FileNotFoundError, ValueError):
                date = ql.TARGET().advance(date, -ql.Period('1D'))
        
        raise FileNotFoundError(f"No market data found for {self.date} or previous {max_retries} business days")


class MktDataManager(QObject):
    """Manager for market data loading in a QThread."""
    
    # Signals
    completed = Signal(dict)  # result
    failed = Signal(str)  # error message
    
    def __init__(self, path_folder: str = None):
        super().__init__()
        self.path_folder = path_folder or r"\\Umilp-p2.cdm.cm-cic.fr\cic-lai-lae-cigogne$\1_Structuration\6_Lexifi\Market_data"
        self.worker = None
        self.thread = None
    
    def retrieve(self, date: ql.Date = None):
        """Load market data in background thread."""
        # Cleanup previous thread if exists
        if self.thread is not None:
            try:
                if self.thread.isRunning():
                    self.thread.quit()
                    self.thread.wait()
            except RuntimeError:
                # C++ object already deleted, just ignore
                pass
            finally:
                self.thread = None
                self.worker = None
        
        # Create new thread and worker
        self.thread = QThread()
        self.worker = MktDataWorker(self.path_folder, date)
        self.worker.moveToThread(self.thread)
        
        # Connect signals
        self.thread.started.connect(self.worker.run)
        self.worker.completed.connect(self._on_completed)
        self.worker.failed.connect(self._on_failed)
        self.worker.completed.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        
        self.thread.start()
    
    @Slot(dict)
    def _on_completed(self, data: dict):
        """Handle successful data load."""
        self.completed.emit(data)
    
    @Slot(Exception)
    def _on_failed(self, error: Exception):
        """Handle data load failure."""
        self.failed.emit(str(error))
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            if hasattr(self, 'thread') and self.thread:
                if self.thread.isRunning():
                    self.thread.quit()
                    if not self.thread.wait(5000):
                        self.thread.terminate()
                        self.thread.wait()
        except (RuntimeError, AttributeError):
            pass


class Task:
    """Represents a single task to be executed."""
    
    def __init__(self, task_id: str, func: Callable, args: tuple = (), kwargs: dict = None, result_dialog=None):
        """
        Initialize a task.
        
        Args:
            task_id: Unique identifier for the task
            func: Callable function to execute
            args: Positional arguments for the function
            kwargs: Keyword arguments for the function
            result_dialog: Dialog to display results for this task
        """
        self.task_id = task_id
        self.func = func
        self.args = args
        self.kwargs = kwargs or {}
        self.result = None
        self.error = None
        self.result_dialog = result_dialog
    
    def execute(self):
        """Execute the task and store result or error."""
        try:
            self.result = self.func(*self.args, **self.kwargs)
        except Exception as e:
            self.error = e
            traceback.print_exc()

class PricingWorker(QObject):
    """Worker that processes tasks from a queue."""
    
    # Signals
    task_started = Signal(str)  # task_id
    task_completed = Signal(str, object)  # task_id, result
    task_failed = Signal(str, Exception)  # task_id, error
    queue_empty = Signal()
    
    def __init__(self):
        super().__init__()
        self.queue = Queue()
        self.is_running = False
    
    def add_task(self, task: Task):
        """Add a task to the queue."""
        self.queue.put(task)
    
    @Slot()
    def process_tasks(self):
        """Process all tasks in the queue."""
        self.is_running = True
        
        while self.is_running:
            try:
                # Get task with timeout to allow stopping
                task = self.queue.get(timeout=1)
            except:
                # Queue is empty
                if self.queue.empty():
                    self.queue_empty.emit()
                continue
            
            # Execute task
            self.task_started.emit(task.task_id)
            task.execute()
            
            # Emit appropriate signal
            if task.error:
                self.task_failed.emit(task.task_id, task.error)
            else:
                self.task_completed.emit(task.task_id, task.result)
            
            self.queue.task_done()
    
    @Slot()
    def stop(self):
        """Stop processing tasks."""
        self.is_running = False

class PriceManager(QObject):
    """Manager for pricing requests running in a QThread."""
    
    # Signals
    task_started = Signal(str)
    task_completed = Signal(str, object)
    task_failed = Signal(str, Exception)
    queue_empty = Signal()
    
    def __init__(self, dialog_class=None):
        super().__init__()
        self.worker = PricingWorker()
        self.thread = QThread()
        self.result_dialog_class = dialog_class
        self.task_dialogs = {}  # Store dialogs for each task
        self.task_count=0
        # Move worker to thread
        self.worker.moveToThread(self.thread)
        
        # Connect signals
        self.thread.started.connect(self.worker.process_tasks)
        self.worker.task_started.connect(self._on_task_started)
        self.worker.task_completed.connect(self._on_task_completed)
        self.worker.task_failed.connect(self._on_task_failed)
        self.worker.queue_empty.connect(self.queue_empty.emit)
    
    @Slot(str)
    def _on_task_started(self, task_id: str):
        """Handle task started - show loading page."""
        self.task_count+=1
        self.task_started.emit(task_id)
        if task_id in self.task_dialogs:
            dialog = self.task_dialogs[task_id]
            dialog.show_loading_page()
    
    @Slot(str, object)
    def _on_task_completed(self, task_id: str, result: object):
        """Handle task completed - show result page."""
        self.task_completed.emit(task_id, result)
        if task_id in self.task_dialogs:
            dialog = self.task_dialogs[task_id]
            dialog.show_table_page(result if isinstance(result, tuple) else (result,))
    
    @Slot(str, Exception)
    def _on_task_failed(self, task_id: str, error: Exception):
        """Handle task failed."""
        self.task_failed.emit(task_id, error)
    
    def start(self):
        """Start the task processing thread."""
        if not self.thread.isRunning():
            self.thread.start()
    
    def stop(self):
        """Stop the task processing thread."""
        self.worker.stop()
        self.thread.quit()
        self.thread.wait()
    
    def add_task(self, task_id: str, func: Callable, args: tuple = (), kwargs: dict = None):
        """
        Add a task to the queue.
        
        Args:
            task_id: Unique identifier for the task
            func: Function to execute
            args: Positional arguments
            kwargs: Keyword arguments
        """
        # Ensure thread is running
        if not self.thread.isRunning():
            self.start()
        
        # Create a new result dialog for this task
        result_dialog = None
        if self.result_dialog_class:
            result_dialog = self.result_dialog_class()
            self.task_dialogs[task_id] = result_dialog
        
        task = Task(task_id, func, args, kwargs, result_dialog)
        self.worker.add_task(task)
    
    def queue_size(self) -> int:
        """Get current queue size."""
        return self.worker.queue.qsize()
    
    def is_queue_empty(self) -> bool:
        """Check if queue is empty."""
        return self.worker.queue.empty()
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            if hasattr(self, 'thread') and hasattr(self, 'worker') and self.thread and self.worker:
                if self.thread.isRunning():
                    self.worker.stop()
                    self.thread.quit()
                    # Wait up to 5 seconds for thread to finish
                    if not self.thread.wait(5000):
                        self.thread.terminate()
                        self.thread.wait()
        except (RuntimeError, AttributeError):
            # C++ object already deleted or attribute missing, nothing to do
            pass
