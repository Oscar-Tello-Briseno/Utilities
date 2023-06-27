import threading
import multiprocessing
import time

class ThreadManager:
    def __init__(self):
        self.lock = threading.Lock()  # A lock for thread synchronization
        self.condition = threading.Condition()  # A condition for thread synchronization
        self.semaphore = multiprocessing.Semaphore()  # A semaphore for process synchronization

    def run_threads(self, thread_count):
        """Uses threading to run multiple tasks concurrently."""
        threads = []

        try:
            for _ in range(thread_count):
                thread = threading.Thread(target=self.task_with_condition)
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()
        except Exception as e:
            print(f"Error in run_threads: {e}")
        finally:
            for thread in threads:
                if thread.is_alive():
                    thread.join()  # Wait for any remaining threads to finish

    def run_processes(self, process_count):
        """Uses multiprocessing to run multiple tasks concurrently."""
        processes = []

        try:
            for _ in range(process_count):
                process = multiprocessing.Process(target=self.task_with_semaphore)
                processes.append(process)
                process.start()

            for process in processes:
                process.join()
        except Exception as e:
            print(f"Error in run_processes: {e}")
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()  # Terminate any remaining processes

    def task_with_condition(self):
        """The task to be executed concurrently by threads with condition."""
        try:
            with self.condition:
                self.condition.wait()  # Wait for condition signal
                with self.lock:
                    # Perform thread-safe operations here
                    print("Executing task with condition...")
        except Exception as e:
            print(f"Error in task with condition: {e}")

    def task_with_semaphore(self):
        """The task to be executed concurrently by processes with semaphore."""
        try:
            self.semaphore.acquire()  # Acquire semaphore
            with self.lock:
                # Perform process-safe operations here
                print("Executing task with semaphore...")
            self.semaphore.release()  # Release semaphore
        except Exception as e:
            print(f"Error in task with semaphore: {e}")

    def signal_condition(self):
        """Signals the condition to wake up waiting threads."""
        try:
            with self.condition:
                self.condition.notify_all()
        except Exception as e:
            print(f"Error in signaling condition: {e}")

    def cleanup(self):
        """Cleans up any resources used by threads or processes."""
        try:
            self.condition.acquire()
            self.condition.notify_all()
        finally:
            self.condition.release()

        try:
            self.semaphore.acquire()
            self.semaphore.release()
        except ValueError:
            pass


if __name__ == "__main__":
    manager = ThreadManager()

    try:
        # Run tasks using threading
        manager.run_threads(3)

        # Simulate some processing time
        time.sleep(2)

        # Signal the condition to wake up waiting threads
        manager.signal_condition()

        # Run tasks using multiprocessing
        manager.run_processes(2)
    finally:
        # Perform clean-up operations
        manager.cleanup()
