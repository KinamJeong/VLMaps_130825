import subprocess
import time

FLAG_FILE = "llm_flag.txt"
TARGET_SCRIPT = "evaluate_object_goal_navigation.py"
CHECK_INTERVAL = 1  # 초 단위

def read_flag():
    try:
        with open(FLAG_FILE, "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        return "0"

def write_flag(value):
    with open(FLAG_FILE, "w") as f:
        f.write(str(value))

def main():
    print("[INFO] Monitoring flag.txt for signal...")
    while True:
        flag = read_flag()
        if flag == "1":
            print("[INFO] Trigger detected! Running evaluation script...")
            try:
                subprocess.run(["python3", TARGET_SCRIPT], check=True)
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] Script failed: {e}")
            finally:
                write_flag("0")
                print("[INFO] Reset flag.txt to 0.")
                break
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()
