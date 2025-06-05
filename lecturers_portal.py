import os
import csv
import pandas as pd
from datetime import datetime
import hashlib
import getpass
import base64
import keyring
from email.message import EmailMessage
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import google.auth.exceptions

class LecturerPortal:
    def __init__(self):
        self.lecturers_file = "lecturers.csv"
        self.courses_file = "courses.csv"
        self.user_details_file = "user_details.csv"
        self.attendance_dir = "attendance_records"
        self.enrollments_file = "course_enrollments.csv"
        self.current_lecturer = None
        self.current_course = None
        self.attendance_session = []
        self.email_config = "email_config.csv"
        self.token_file = "gmail_token.json"
        self.credentials_file = "gmail_credentials.json"
        self.SCOPES = ['https://www.googleapis.com/auth/gmail.send']
    
        # Create files if they don't exist
        self.initialize_files()
    
    def initialize_files(self):
        """Create necessary files if they don't exist"""
        # Create enrollments file with header if it doesn't exist
        if not os.path.exists(self.enrollments_file):
            with open(self.enrollments_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["CourseCode", "StudentID", "EnrollmentType"])
        
    # Define Scope    
    SCOPES = ['https://www.googleapis.com/auth/gmail.send']
 
    def hash_password(self, password):
        return hashlib.sha256(password.encode()).hexdigest()

    def lecturer_login(self):
        print("\n" + "="*20 + " LECTURER LOGIN " + "="*20)
        email = input("Email: ").strip()
        password = getpass.getpass("Password: ")
        hashed_password = self.hash_password(password)
        
        try:
            with open(self.lecturers_file, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["Email"] == email and row["Password"] == hashed_password:
                        self.current_lecturer = row
                        print(f"\nWelcome, {row['Name']}!")
                        return True
        except Exception as e:
            print(f"Login error: {str(e)}")
        
        print("Invalid email or password!")
        return False

    def get_assigned_courses(self):
        """Get courses assigned to the current lecturer"""
        courses = []
        try:
            with open(self.courses_file, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["LecturerID"] == self.current_lecturer["LecturerID"]:
                        courses.append(row)
        except Exception as e:
            print(f"Error reading courses: {str(e)}")
        return courses

    def select_course(self):
        """Show assigned courses and let lecturer select one"""
        courses = self.get_assigned_courses()
        if not courses:
            print("No courses assigned to you!")
            return False
            
        print("\n" + "="*20 + " YOUR COURSES " + "="*20)
        for i, course in enumerate(courses, 1):
            print(f"{i}. {course['CourseName']} ({course['CourseCode']}) - {course['Session']}")
        
        try:
            choice = int(input("\nSelect a course: ").strip())
            if 1 <= choice <= len(courses):
                self.current_course = courses[choice-1]
                return True
        except:
            pass
            
        print("Invalid selection!")
        return False

    #def start_attendance_session(self):
        """Mark attendance for the selected course"""
        if not self.current_course:
            print("No course selected!")
            return
            
        print(f"\nStarting attendance session for {self.current_course['CourseName']}...")
        print("Press 'q' to stop attendance session")
        
        # Import face attendance system only when needed
        from facial_attendance import FaceAttendanceSystem
        
        system = FaceAttendanceSystem()
        system.mark_attendance_for_course(
            self.current_course['CourseCode'],
            self.current_course['CourseName'],
            self.current_course['Session']
        )

    def view_attendance(self):
        """View attendance records for the selected course"""
        if not self.current_course:
            print("No course selected!")
            return
            
        course_code = self.current_course["CourseCode"]
        session = self.current_course["Session"]
        
        print(f"\nAttendance for {self.current_course['CourseName']} ({course_code} - {session})")
        
        # Get all attendance files for this course
        attendance_files = []
        for filename in os.listdir(self.attendance_dir):
            if filename.startswith(f"attendance_{course_code}_") and session in filename:
                attendance_files.append(filename)
        
        if not attendance_files:
            print("No attendance records found!")
            return
            
        # Show all attendance sessions
        print("\nAttendance Sessions:")
        for i, filename in enumerate(attendance_files, 1):
            date_str = filename.split('_')[2].split('.')[0]
            print(f"{i}. {date_str}")
        
        try:
            choice = int(input("\nSelect a session: ").strip())
            if 1 <= choice <= len(attendance_files):
                selected_file = attendance_files[choice-1]
                filepath = os.path.join(self.attendance_dir, selected_file)
                
                # Read and display attendance records
                df = pd.read_csv(filepath)
                print("\nAttendance Records:")
                print(df[['ID','Name', 'Time']])
        except:
            print("Invalid selection!")

    def generate_attendance_report(self):
        """Generate Excel report for selected course"""
        if not self.current_course:
            print("No course selected!")
            return
            
        course_code = self.current_course["CourseCode"]
        session = self.current_course["Session"]
        
        # Get all attendance files for this course
        attendance_files = []
        for filename in os.listdir(self.attendance_dir):
            if filename.startswith(f"attendance_{course_code}_") and session in filename:
                attendance_files.append(os.path.join(self.attendance_dir, filename))
        
        if not attendance_files:
            print("No attendance records found!")
            return
            
        # Combine all attendance records
        all_records = []
        for filepath in attendance_files:
            df = pd.read_csv(filepath)
            all_records.append(df)
        
        combined_df = pd.concat(all_records, ignore_index=True)
        
        # Pivot to create student attendance matrix
        pivot_df = combined_df.pivot_table(
            index=['ID', 'Name'],
            columns='Time',
            values='Confidence',
            aggfunc='first'
        ).reset_index()
        
        # Generate report filename
        report_filename = f"attendance_report_{course_code}_{session}_{datetime.now().strftime('%Y%m%d')}.xlsx"
        report_path = os.path.join(self.attendance_dir, report_filename)
        
        # Save to Excel
        pivot_df.to_excel(report_path, index=False)
        print(f"\nReport generated: {report_filename}")
        print(f"Saved to: {os.path.abspath(report_path)}")
    
    def configure_email_settings(self):
        """Configure email settings for the lecturer using yagmail"""
        print("\n" + "="*20 + " EMAIL CONFIGURATION " + "="*20)
        print("Configure your email settings to send attendance reports")
        
        # Check if configuration already exists
        config_exists = False
        lecturer_id = self.current_lecturer["LecturerID"]
        config_data = {}
        
        if os.path.exists(self.email_config):
            with open(self.email_config, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["LecturerID"] == lecturer_id:
                        config_exists = True
                        config_data = row
                        break
        
        if config_exists:
            print(f"\nCurrent settings:")
            print(f"Email: {config_data['Email']}")
            change = input("\nDo you want to change these settings? (y/n): ").strip().lower()
            if change != 'y':
                return
        
        email = input("Your email address: ").strip()
        
        # Prepare new config
        new_config = {
            "LecturerID": lecturer_id,
            "Email": email
        }
        
        # Save configuration
        fieldnames = ["LecturerID", "Email"]
        file_exists = os.path.exists(self.email_config)
        
        with open(self.email_config, "a" if file_exists else "w", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            if config_exists:
                # Update existing config
                self.update_email_config(lecturer_id, new_config)
            else:
                writer.writerow(new_config)
        
        # Store password securely using keyring
        password = getpass.getpass("Email password (for sending emails): ")
        keyring.set_password("lecturer_portal", email, password)
        
        print("\nEmail settings saved securely!")

    def update_email_config(self, lecturer_id, new_config):
        """Update existing email configuration"""
        rows = []
        with open(self.email_config, "r") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            for row in reader:
                if row["LecturerID"] == lecturer_id:
                    rows.append(new_config)
                else:
                    rows.append(row)
        
        with open(self.email_config, "w", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def get_email_config(self):
        """Retrieve email configuration for current lecturer"""
        lecturer_id = self.current_lecturer["LecturerID"]
        
        if os.path.exists(self.email_config):
            with open(self.email_config, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["LecturerID"] == lecturer_id:
                        return row
        return None

    def get_student_emails(self):
        """Retrieve student emails from user_details.csv"""
        students_file = "user_details.csv"
        emails = {}
        
        if os.path.exists(students_file):
            with open(students_file, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    user_id = row["UserID"]
                    email = row["Email"]
                    emails[user_id] = email
        return emails

    def get_attendance_summary(self):
        """Get attendance summary for the current course"""
        course_code = self.current_course["CourseCode"]
        session = self.current_course["Session"]
        
        # Get all attendance files
        attendance_files = []
        for filename in os.listdir(self.attendance_dir):
            if filename.startswith(f"attendance_{course_code}_") and session in filename:
                attendance_files.append(os.path.join(self.attendance_dir, filename))
        
        if not attendance_files:
            print("No attendance records found!")
            return None
        
        # Combine attendance records
        all_records = []
        for filepath in attendance_files:
            df = pd.read_csv(filepath)
            # Extract date from filename
            date_str = os.path.basename(filepath).split('_')[2].split('.')[0]
            df['Date'] = date_str
            all_records.append(df)
        
        combined_df = pd.concat(all_records, ignore_index=True)
        
        # Calculate attendance stats
        attendance_stats = combined_df.groupby(['ID', 'Name'])['Date'].agg(
            Total_Sessions='count',
            Sessions_Attended=lambda x: len(x.unique())
        ).reset_index()
        
        total_sessions = len(attendance_files)
        attendance_stats['Attendance_Percentage'] = (attendance_stats['Sessions_Attended'] / total_sessions) * 100
        
        return attendance_stats, total_sessions
    
    
    def get_gmail_service(self):
        """Authenticate and create Gmail API service"""
        creds = None
        # Load existing token if available
        if os.path.exists(self.token_file):
            try:
                creds = Credentials.from_authorized_user_file(self.token_file, self.SCOPES)
            except (ValueError, json.JSONDecodeError) as e:
                print(f"Error loading token: {str(e)}")
                os.remove(self.token_file)  # Remove invalid token
        
        # If token is expired or doesn't exist, refresh or get new token
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                except google.auth.exceptions.RefreshError:
                    creds = self.obtain_new_token()
            else:
                creds = self.obtain_new_token()
            
            # Save the token for next time
            with open(self.token_file, 'w') as token:
                token.write(creds.to_json())
        
        try:
            service = build('gmail', 'v1', credentials=creds)
            return service
        except Exception as e:
            print(f"Failed to create Gmail service: {str(e)}")
            return None

    def obtain_new_token(self):
        """Obtain new OAuth token through user consent"""
        if not os.path.exists(self.credentials_file):
            print("\n" + "="*50)
            print("GOOGLE OAUTH SETUP REQUIRED")
            print("="*50)
            print("1. Go to Google Cloud Console: https://console.cloud.google.com/")
            print("2. Create a project and enable Gmail API")
            print("3. Create OAuth 2.0 credentials (Desktop App type)")
            print("4. Download credentials JSON file and save as 'gmail_credentials.json'")
            print("5. Place it in the same directory as this application")
            input("\nPress Enter after completing these steps...")
        
        if not os.path.exists(self.credentials_file):
            print("Credentials file not found! Please create 'gmail_credentials.json'")
            return None
        
        try:
            flow = InstalledAppFlow.from_client_secrets_file(
                self.credentials_file, self.SCOPES)
            creds = flow.run_local_server(port=0)
            return creds
        except Exception as e:
            print(f"Authentication failed: {str(e)}")
            return None
    
    def send_email_gmail_api(self, to_email, subject, body):
        """Send email using Gmail API"""
        try:
            service = self.get_gmail_service()
            if not service:
                return False
            
            message = EmailMessage()
            message.set_content(body)
            message['To'] = to_email
            message['From'] = self.current_lecturer['Email']
            message['Subject'] = subject
            
            message_bytes = message.as_bytes()
            
            encoded_message = base64.urlsafe_b64encode(message_bytes).decode()
            
            create_message = {'raw': encoded_message}
            
            send_message = service.users().messages().send(
                userId="me", body=create_message).execute()
            return True
        except Exception as e:
            print(f"Failed to send email: {str(e)}")
            return False

    def send_attendance_email(self):
        """Send attendance summary email to students using Gmail API"""
        if not self.current_course:
            print("No course selected!")
            return
        
        # Get attendance summary
        attendance_stats, total_sessions = self.get_attendance_summary()
        if attendance_stats is None:
            return
        
        # Get student emails
        student_emails = self.get_student_emails()
        if not student_emails:
            print("No student emails found!")
            return
        
        course_name = self.current_course["CourseName"]
        lecturer_name = self.current_lecturer["Name"]
        lecturer_email = self.current_lecturer["Email"]
        
        # Verify we have lecturer email
        if not lecturer_email:
            print("Lecturer email not found in profile!")
            return
        
        print(f"\nPreparing to send attendance emails for {course_name}...")
        print(f"Lecturer: {lecturer_name} <{lecturer_email}>")
        
        # Send emails to each student
        success_count = 0
        total_students = len(attendance_stats)
        
        if total_students == 0:
            print("No attendance records to send!")
            return
            
        for _, row in attendance_stats.iterrows():
            student_id = row['ID']
            if student_id not in student_emails:
                print(f"No email found for student ID: {student_id}")
                continue
                
            student_email = student_emails[student_id]
            student_name = row['Name']
            
            # Create email content
            subject = f"Attendance Summary for {course_name}"
            
            body = f"""
Dear {student_name},

Here is your attendance summary for the course:
Course: {course_name} ({self.current_course['CourseCode']})
Session: {self.current_course['Session']}
Lecturer: {lecturer_name}

- Total Sessions: {total_sessions}
- Sessions Attended: {row['Sessions_Attended']}
- Attendance Percentage: {row['Attendance_Percentage']:.2f}%

If you have any questions about your attendance record, please contact your lecturer.

Best regards,
{lecturer_name}
{lecturer_email}
            """
            
            # Send email using Gmail API
            try:
                success = self.send_email_gmail_api(student_email, subject, body)
                if success:
                    print(f"✓ Email sent to {student_name} at {student_email}")
                    success_count += 1
                else:
                    print(f"✗ Failed to send email to {student_email}")
            except Exception as e:
                print(f"✗ Error sending to {student_email}: {str(e)}")
        
        print(f"\nEmail sending summary:")
        print(f"- Attempted to send: {total_students} emails")
        print(f"- Successfully sent: {success_count} emails")
        print(f"- Failed to send: {total_students - success_count} emails")
        
        if total_students - success_count > 0:
            print("\nTroubleshooting tips:")
            print("1. Ensure you have 'gmail_credentials.json' from Google Cloud Console")
            print("2. Make sure you've enabled Gmail API for your project")
            print("3. Verify the credentials file is in the correct format")
            print("4. Check that your Google account has permission to send emails")
    
    
    def get_course_students(self):
        """Get students enrolled in the current course"""
        if not self.current_course:
            return []
            
        course_code = self.current_course["CourseCode"]
        session = self.current_course["Session"]
        
        # Extract department and level from course code
        try:
            department = course_code[:3].upper()  # Normalize to uppercase
            # Extract level from course code (e.g., "401" -> 400)
            level_num = int(course_code[3:])
            level = str((level_num // 100) * 100)
        except:
            department = ""
            level = ""
        
        enrolled_students = []
        student_count = 0
        
        # Debug output
        print(f"Course: {course_code}, Seeking Dept: {department}, Level: {level}")
        
        if os.path.exists(self.user_details_file):
            with open(self.user_details_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    student_count += 1
                    try:
                        # Get matriculation number
                        matric = row.get('UserID', '').strip()
                        
                        # Skip if matriculation is empty
                        if not matric:
                            continue
                            
                        # Extract year part - more flexible approach
                        year_part = ""
                        if len(matric) >= 4:
                            # Try different patterns
                            if matric[0:2].isalpha() and matric[2:4].isdigit():
                                year_part = matric[2:4]  # Pattern: AB21XXX
                            elif matric[0:2].isdigit():
                                year_part = matric[0:2]  # Pattern: 21XXXX
                            elif matric[-4:-2].isdigit():
                                year_part = matric[-4:-2]  # Pattern: XXX21XX
                        
                        if not year_part or not year_part.isdigit():
                            continue
                        
                        admission_year = 2000 + int(year_part)
                        current_year = datetime.now().year
                        years_since_admission = current_year - admission_year
                        
                        # Adjust level calculation based on academic system
                        # This assumes 1st year = 200, 2nd year = 300, etc.
                        student_level = str((years_since_admission + 1) * 100)
                        
                        # Get department (first 3 chars, uppercase)
                        student_dept = row.get('Department', '').strip().upper()

                        
                        # Debug output per student
                        debug_info = f"Student: {row.get('Name', '')} ({row.get('UserID', '')}) "
                        debug_info += f"Matric: {matric}, Dept: {student_dept}, Level: {student_level}"
                        
                        # Check match
                        if student_dept == department and student_level == level:
                            enrolled_students.append({
                                'ID': row['UserID'],
                                'Name': row['Name'],
                                'Department': row['Department'],
                                'Level': student_level,
                                'EnrollmentType': 'Regular'
                            })
                            print(f"MATCHED: {debug_info}")
                        else:
                            print(f"NOT MATCHED: {debug_info}")
                    except Exception as e:
                        print(f"Error processing student {row.get('UserID', '')}: {str(e)}")
        
        # Add carryover students from enrollments file
        if os.path.exists(self.enrollments_file):
            with open(self.enrollments_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["CourseCode"] == course_code:
                        # Find student in user_detail
                        student = self.get_student_by_id(row["StudentID"])
                        if student:
                            enrolled_students.append({
                                'ID': student['UserID'],
                                'Name': student['Name'],
                                'Department': student['Department'],
                                'Level': student.get('Level', ''),
                                'EnrollmentType': 'Carryover'
                            })
        
        # Debug summary
        print(f"Total students in DB: {student_count}")
        print(f"Matched regular students: {len([s for s in enrolled_students if s['EnrollmentType'] == 'Regular'])}")
        print(f"Carryover students: {len([s for s in enrolled_students if s['EnrollmentType'] == 'Carryover'])}")
        
        return enrolled_students
          
    def get_student_by_id(self, student_id):
        """Get student details by ID"""
        if os.path.exists(self.user_details_file):
            with open(self.user_details_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['UserID'] == student_id:
                        return row
        return None

    def view_course_students(self):
        """View students enrolled in the current course"""
        
        if self.current_course is None or not isinstance(self.current_course, dict):
           print("No course selected!")
           return
       
        course_code = self.current_course["CourseCode"]
        session = self.current_course["Session"]
            
        students = self.get_course_students()
        if not students:
            print("No students enrolled in this course!")
            return
            
        print(f"\nStudents enrolled in {self.current_course['CourseName']} ({self.current_course['CourseCode']}):")
        print("=" * 80)
        print(f"{'ID':<10} {'Name':<25} {'Department':<15} {'Level':<8} {'Type':<10}")
        print("-" * 80)
        for student in students:
            print(f"{student['ID']:<10} {student['Name']:<25} {student['Department']:<15} {student['Level']:<8} {student['EnrollmentType']:<10}")
        print("=" * 80)
        print(f"Total students: {len(students)}")
        return students
    
    def add_student_to_course(self):
        """Add a student to the current course (for carryovers)"""
        if not self.current_course:
            print("No course selected!")
            return
            
        student_id = input("Enter student ID to add: ").strip()
        if not student_id:
            print("Invalid student ID!")
            return
            
        # Check if student exists
        student = self.get_student_by_id(student_id)
        if not student:
            print(f"Student with ID {student_id} not found!")
            return
            
        # Check if already enrolled
        current_students = self.get_course_students()
        if any(s['ID'] == student_id for s in current_students):
            print(f"Student {student_id} is already enrolled in this course!")
            return
            
        # Add to enrollments
        with open(self.enrollments_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.current_course["CourseCode"],
                student_id,
                "Carryover"
            ])
            
        print(f"Student {student_id} added to the course successfully!")

    def take_manual_attendance(self):
        """Take attendance manually by marking students present/absent"""
        if not self.current_course:
            print("No course selected!")
            return
            
        students = self.get_course_students()
        if not students:
            print("No students enrolled in this course!")
            return
            
        # Create attendance record
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"attendance_{self.current_course['CourseCode']}_{timestamp}.csv"
        filepath = os.path.join(self.attendance_dir, filename)
        
        # Create attendance directory if needed
        os.makedirs(self.attendance_dir, exist_ok=True)
        
        # Prepare attendance data
        attendance_data = []
        print("\n" + "="*50)
        print(f"MANUAL ATTENDANCE FOR {self.current_course['CourseName']}")
        print("="*50)
        
        for student in students:
            status = input(f"Is {student['Name']} ({student['ID']}) present? (y/n): ").strip().lower()
            attendance_data.append({
                'ID': student['ID'],
                'Name': student['Name'],
                'Status': 'Present' if status == 'y' else 'Absent',
                'Time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'CourseCode': self.current_course['CourseCode'],
                'CourseName': self.current_course['CourseName'],
                'Session': self.current_course['Session']
            })
        
        # Save to CSV
        with open(filepath, 'w', newline='') as f:
            fieldnames = ['ID', 'Name', 'Status', 'Time', 'CourseCode', 'CourseName', 'Session']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(attendance_data)
            
        print(f"\nAttendance recorded successfully for {len(students)} students!")
        print(f"Saved to: {filepath}")

    def view_attendance_report(self):
        """View attendance report for the current course"""
        if not self.current_course:
            print("No course selected!")
            return

        course_code = self.current_course["CourseCode"]
        session = self.current_course["Session"]

        # Step 1: Find ALL attendance files for this course
        attendance_files = []
        for filename in os.listdir(self.attendance_dir):
            # Match pattern: attendance_{course_code}_{date}.csv
            if filename.startswith(f"attendance_{course_code}_") and filename.endswith(".csv"):
                # Extract date from filename
                date_str = filename.split('_')[2].split('.')[0]
                # Store both filename and its date
                attendance_files.append((filename, date_str))

        if not attendance_files:
            print("No attendance records found!")
            return

        # Sort by date so sessions appear chronologically
        attendance_files.sort(key=lambda x: x[1])

        # Step 2: Load all attendance records
        all_records = []
        unique_dates = set()
        for filename, date_str in attendance_files:
            try:
                file_path = os.path.join(self.attendance_dir, filename)
                df = pd.read_csv(file_path)
                # Add session date column
                df['SessionDate'] = date_str
                df['Status'] = 'Present'
                all_records.append(df)
                unique_dates.add(date_str)
            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")

        if not all_records:
            print("No valid attendance records found!")
            return

        combined_df = pd.concat(all_records, ignore_index=True)

        # Step 3: Get enrolled students
        students = self.get_course_students()
        if not students:
            print("No students enrolled in this course!")
            return
        student_df = pd.DataFrame(students)

        # Step 4: Create complete student-date matrix
        # Use sorted unique dates for chronological order
        attendance_dates = sorted(unique_dates)
        student_df['key'] = 1
        dates_df = pd.DataFrame({'SessionDate': attendance_dates, 'key': 1})
        full_matrix = student_df.merge(dates_df, on='key').drop('key', axis=1)

        # Step 5: Merge with attendance data
        merged_df = full_matrix.merge(
            combined_df[['ID', 'SessionDate', 'Status']],
            on=['ID', 'SessionDate'],
            how='left',
            suffixes=('', '_y')
        )
        
        # Step 6: Clean status (Present/Absent)
        merged_df['Status'] = merged_df['Status'].fillna('Absent')
        merged_df = merged_df[['ID', 'Name', 'SessionDate', 'Status']]

        # Step 7: Create pivot table
        pivot_df = merged_df.pivot_table(
            index=['ID', 'Name'],
            columns='SessionDate',
            values='Status',
            aggfunc='first',
            fill_value='Absent'
        ).reset_index()

        # Step 8: Calculate statistics
        date_columns = [col for col in pivot_df.columns if col not in ['ID', 'Name']]
        total_sessions = len(date_columns)

        pivot_df['PresentCount'] = pivot_df[date_columns].apply(
            lambda row: (row == 'Present').sum(),
            axis=1
        )
        
        pivot_df['AttendancePercentage'] = (pivot_df['PresentCount'] / total_sessions) * 100

        # Step 9: Display report
        print("\n" + "=" * 80)
        print(f"ATTENDANCE REPORT: {self.current_course['CourseName']} ({course_code})")
        print("=" * 80)
        print(f"Total Sessions: {total_sessions}")
        print(pivot_df)

        # Step 10: Save report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_filename = f"attendance_report_{course_code}_{timestamp}.xlsx"
        report_path = os.path.join(self.attendance_dir, report_filename)

        try:
            pivot_df.to_excel(report_path, index=False)
            print(f"\nReport saved to: {report_path}")
        except Exception as e:
            print(f"\nError saving report: {str(e)}")
            print("Make sure the file is not open in another program")

    def lecturer_menu(self):
        while True:
            print("\n" + "="*20 + " LECTURER PORTAL " + "="*20)
            if self.current_course:
                print(f"Current Course: {self.current_course['CourseName']} ({self.current_course['CourseCode']})")
            
            print("1. Select Course")
            print("2. Start Facial Attendance Session")
            print("3. Take Manual Attendance")
            print("4. View Course Students")
            print("5. Add Student to Course (Carryover)")
            print("6. View Attendance Records")
            print("7. Generate Attendance Report")
            print("8. Configure Email Settings")
            print("9. Send Attendance Email to Students")
            print("10. Switch Lecturer")
            print("11. Exit")
            
            choice = input("Select option: ").strip()
            
            if choice == "1":
                self.select_course()
            elif choice == "2":
                self.start_attendance_session()
            elif choice == "3":
                self.take_manual_attendance()
            elif choice == "4":
                self.view_course_students()
            elif choice == "5":
                self.add_student_to_course()
            elif choice == "6":
                self.view_attendance()
            elif choice == "7":
                self.view_attendance_report()
            elif choice == "8":
                self.configure_email_settings()
            elif choice == "9":
                self.send_attendance_email()
            elif choice == "10":
                self.current_lecturer = None
                self.current_course = None
                if self.lecturer_login():
                    self.select_course()
            elif choice == "11":
                print("Exiting lecturer portal...")
                break
            else:
                print("Invalid choice!")

    def start_attendance_session(self):
        """Mark attendance for the selected course with facial recognition"""
        if not self.current_course:
            print("No course selected!")
            return
            
        print(f"\nStarting attendance session for {self.current_course['CourseName']}...")
        print("Press 'q' to stop attendance session")
        
        # Import face attendance system only when needed
        from facial_attendance import FaceAttendanceSystem
        
        # Get course students for recognition
        course_students = self.get_course_students()
        if not course_students:
            print("No students enrolled in this course!")
            return
            
        # Create student ID list for facial recognition
        student_ids = [s['ID'] for s in course_students]
        
        system = FaceAttendanceSystem()
        system.student_ids = student_ids # Pass enrolled student IDs
        system.mark_attendance_for_course(
            self.current_course['CourseCode'],
            self.current_course['CourseName'],
            self.current_course['Session'],
           
        )

if __name__ == "__main__":
    portal = LecturerPortal()
    if portal.lecturer_login():
        portal.select_course()
        portal.lecturer_menu()