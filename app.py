import psutil
import time

def monitor_ram_usage():
    process = psutil.Process()
    while True:
        memory_info = process.memory_info()
        ram_usage = memory_info.rss / 1024 / 1024  # Convert to MB
        print(f"RAM usage: {ram_usage:.2f} MB")
        time.sleep(1)  # Wait for 1 second

if __name__ == "__main__":
    monitor_ram_usage()
