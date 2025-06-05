import csv
import os

from facial_attendance import FaceAttendanceSystem

if __name__ == "__main__":
    system = FaceAttendanceSystem()
    
    # Check if admin exists
    admin_exists = False
    if os.path.exists(system.user_detail_file):
        try:
            with open(system.user_detail_file, "r") as f:
                reader = csv.DictReader(f)
                admin_exists = any(row['UserID'] == 'admin' for row in reader)
        except:
            pass
            
    if not admin_exists:
        print("No admin found. Creating initial admin account...")
        system.create_admin()
    
    system.main_menu()