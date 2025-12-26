
from PySide6.QtCore import QObject,Signal,Slot,QThread
from queue import Queue

import os
import QuantLib as ql
import pandas as pd
import numpy as np
from typing import Callable
import traceback

from Pricing.Equity import EQDataPrep
from Pricing.Equity.Model import HestonModel

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
        self.path_folder = path_folder or r"C:\Users\jorda\OneDrive\Documents\pricer_interface-main\snapshot"
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


##Equity part
def get_callprobas(S_fix, call_lvl, S0):
    Nsim, n = S_fix.shape
    probas = np.zeros(n)
    redeemed = []
    for i in range(n - 1):
        idx_call = np.where(S_fix[:,i] >= call_lvl*S0)[0]
        idx_call = np.setdiff1d(idx_call, redeemed)
        probas[i] = len(idx_call)/Nsim
        redeemed = np.union1d(redeemed, idx_call)
    probas[-1] = 1 - np.sum(probas, axis = 0)
    return probas

def geteq_fund2(T_pay, probas, fund_curve, discount):
    c=0.0004
    tenor_spread = fund_curve(T_pay)
    fund2 = (np.sum(tenor_spread*probas, axis = 0) - c)*discount
    return tenor_spread, fund2


class EquityWorker(QObject):
    finished=Signal(dict)

    def __init__(self,input_data:dict):   
        super().__init__()
        self.param=input_data.copy()
        
        self.param["MC"] = 1/365, 10000
        value_date = ql.TARGET().advance(ql.Date.todaysDate(), -1, ql.Days)
        self.param["value_date"] = value_date
        if input_data['currency']=="EUR":
            spread_data = pd.read_excel(os.path.join(EQDataPrep.spread_path, "Refi_CIC_EUR.xls"),sheet_name="CIC_EUR")
        elif input_data['currency']=="USD":
            spread_data=pd.read_excel(os.path.join(EQDataPrep.spread_path, "Refi_CIC_EUR.xls"),sheet_name="CIC_USD")
        else:
            raise ValueError(" Unavailable currency")
        self.param["fund_curve"] = EQDataPrep.spread_prep(spread_data)
        path_markit_names=r"\\Umilp-p2.cdm.cm-cic.fr\cic-lai-lae-cigogne$\1_Structuration\19_Quant\Methodo Funding\Markit\Names.xlsx"
        self.param["ref_table"] = pd.read_excel(path_markit_names, header = 0)


    def compute_result(self):
        dt, Nsim = self.param["MC"]
        value_date = self.param["value_date"]
        fund_curve = self.param["fund_curve"]
        ref_table = self.param["ref_table"]

        #convert param
        start_date = ql.Date(self.param["issue_date"], "%d.%m.%Y")
        trade_date = ql.Date(self.param["initial_strike_date"], "%d.%m.%Y")
        mat = int(self.param["maturity"])
        call_lvl = float(self.param["autocall_level"])*0.01

        freq_dic={'Annually':'1Y','Semi-annually':'6M',
            'Quarterly':'3M','Monthly':'1M'}
        freq = freq_dic[self.param["frequency"]]
        per_nocall = int(self.param["periods_no_autocall"])
        offset = int(self.param["fixing_offset"])

        pay_dates = list(ql.Schedule(start_date, start_date + ql.Period(mat, ql.Years), ql.Period(freq), ql.TARGET(), ql.Following, ql.Following, ql.DateGeneration.Forward, False))[1 + per_nocall:]
        fix_dates = list(map(lambda date: ql.TARGET().advance(date, offset, ql.Days), pay_dates))
        T_pay = np.array(list(map(lambda date: ql.Actual365Fixed().yearFraction(value_date, date), pay_dates)))
        T_fix = np.array(list(map(lambda date: ql.Actual365Fixed().yearFraction(value_date, date), fix_dates)))

        stock = self.param["underlying"]
        vol_surface, forward, df = EQDataPrep.prep_data_markit(value_date.ISO(), EQDataPrep.markit_path, stock, ref_table)

        params = HestonModel.calib_heston(value_date.ISO(), vol_surface)

        Xt = HestonModel.diffuse_heston_1D(params, dt, Nsim, T_fix[-1], 10)
        idx_fix = np.array(T_fix/dt, dtype = int)
        S_fix = forward(T_fix)*Xt[:, idx_fix]
        T_strike = ql.Actual365Fixed().yearFraction(value_date, trade_date)
        S0 = forward(T_strike)
        probas = get_callprobas(S_fix, call_lvl, S0)

        if start_date> value_date+ql.Period('2M'):
            discount=0.95
        else:
            discount=1
        duration = np.sum(T_pay*probas, axis = 0)
        new_spreads, new_fund = geteq_fund2(T_pay, probas, fund_curve, discount)

        res= {"duration": duration,
                "Payment Dates":pay_dates,
                "Early Redemption Proba": probas,
                "Forwards":forward(T_pay),
                'Zero Coupon': df(T_pay), 
                "funding_spread": new_fund}
        
        self.finished.emit(res)



