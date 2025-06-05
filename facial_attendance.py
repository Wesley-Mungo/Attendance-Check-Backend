import cv2
import os
import csv
import numpy as np
from datetime import datetime
import time
from keras_facenet import FaceNet
from mtcnn.mtcnn import MTCNN
import getpass
import pandas as pd
import hashlib

class FaceAttendanceSystem:
    def __init__(self):
        self.detector = MTCNN()
        self.embedder = FaceNet()
        self.EMBEDDING_DIM = 512  # FaceNet produces 512D embeddings

        self.face_threshold = 0.6
        self.required_shots = 10
        self.current_course = None
        
        self.embeddings_file = "face_embeddings.csv"
        self.user_details_file = "user_details.csv"
        self.user_detail_file = "user_detail.csv"
        self.lecturers_file = "lecturers.csv"
        self.courses_file = "courses.csv"
        self.attendance_dir = "attendance_records"
        self.course_enrollments_file = "course_enrollments.csv"
        
        self.student_ids = [] 
        
        os.makedirs(self.attendance_dir, exist_ok=True)
        self.initialize_data_files()

    def initialize_data_files(self):
        # Create embeddings file if not exists
        if not os.path.exists(self.embeddings_file):
            with open(self.embeddings_file, "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["UserID"] + [f"e{i}" for i in range(self.EMBEDDING_DIM)])
        
        # Create user details file if not exists
        if not os.path.exists(self.user_details_file):
            with open(self.user_details_file, "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["UserID", "Name", "Department", "Password", "Email"])
        
        # Create lecturers file if not exists
        if not os.path.exists(self.lecturers_file):
            with open(self.lecturers_file, "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["LecturerID", "Name", "Department", "Email", "Password"])
        
        # Create courses file if not exists
        if not os.path.exists(self.courses_file):
            with open(self.courses_file, "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["CourseCode", "CourseName", "Session", "LecturerID"])

    def hash_password(self, password):
        return hashlib.sha256(password.encode()).hexdigest()

    def authenticate_admin(self):
        print("\n" + "="*20 + " ADMIN LOGIN " + "="*20)
        try:
            with open(self.user_detail_file, "r") as f:
                reader = csv.DictReader(f)
                admins = {row['UserID']: row['Password'] for row in reader if row['UserID'].lower() == 'admin'}

            for attempt in range(3):
                password = getpass.getpass("Enter Admin Password: ")
                if admins.get("admin") == password:
                    return True
                print(f"Invalid password. {2-attempt} attempts remaining.")
            return False

        except Exception as e:
            print(f"Authentication error: {str(e)}")
            return False

   # def create_admin(self):
        print("\n" + "="*20 + " INITIAL ADMIN SETUP " + "="*20)
        password = getpass.getpass("Set Admin Password: ")
        with open(self.user_detail_file, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["admin", "System Admin", "Administration", password, "admin@system"])
        return True

    def align_face(self, image, face):
        """Simple face cropping without alignment"""
        try:
            x, y, w, h = face['box']
            x, y = max(0, x), max(0, y)
            face_img = image[y:y+h, x:x+w]
            return cv2.resize(face_img, (160, 160)) if face_img.size > 0 else None
        except Exception as e:
            print(f"Simple alignment error: {str(e)}")
            return None
       
    def register_user(self):
        print("\n" + "="*20 + " REGISTRATION TYPE " + "="*20)
        print("1. Live Camera Registration")
        print("2. Image Upload Registration")
        choice = input("Select registration method (1/2): ").strip()

        if choice == '1':
            self.live_registration()
        elif choice == '2':
            self.image_upload_registration()
        else:
            print("Invalid choice!")

    def live_registration(self):
        user_id = input("Enter User ID: ").strip()
        name = input("Full Name: ").strip()
        dept = input("Department: ").strip()
        email = input("Email: ").strip()
        password = getpass.getpass("Set Password: ")
        hashed_password = self.hash_password(password)

        cap = cv2.VideoCapture(0)
        embeddings = []
        shot_count, attempt_count = 0, 0
        max_attempts = 100

        while shot_count < self.required_shots and attempt_count < max_attempts:
            ret, frame = cap.read()
            if not ret:
                attempt_count += 1
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = self.detector.detect_faces(rgb)

            if len(faces) == 0:
                cv2.putText(frame, "Face the camera", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            else:
                try:
                    face = max(faces, key=lambda r: r['box'][2] * r['box'][3])
                    aligned = self.align_face(rgb, face)
                    
                    if aligned is None:
                        display_text = "Alignment failed"
                        color = (0, 0, 255)
                    else:
                        # Resize to expected input size
                        aligned = cv2.resize(aligned, (160, 160))
                        
                        embedding = self.embedder.embeddings([aligned])[0]
                        
                        # Verify embedding dimension
                        if len(embedding) != self.EMBEDDING_DIM:
                            print(f"Error: Expected {self.EMBEDDING_DIM}D embedding, got {len(embedding)}D")
                            continue
                        
                        embeddings.append(embedding)
                        shot_count += 1
                        display_text = f"Captured: {shot_count}/{self.required_shots}"
                        color = (0, 255, 0)
                    
                    # Draw face box
                    x, y, w, h = face['box']
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                except Exception as e:
                    display_text = f"Error: {str(e)}"
                    color = (0, 0, 255)

                cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.imshow("Registration", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            attempt_count += 1

        cap.release()
        cv2.destroyAllWindows()

        self.save_registration(user_id, name, dept, hashed_password, email, embeddings)

    def image_upload_registration(self):
        user_id = input("Enter User ID: ").strip()
        name = input("Full Name: ").strip()
        dept = input("Department: ").strip()
        email = input("Email: ").strip()
        password = getpass.getpass("Set Password: ")
        hashed_password = self.hash_password(password)
        folder = input("Enter image folder path: ").strip()

        embeddings = []
        valid_extensions = ('.jpg', '.jpeg', '.png')

        for filename in sorted(os.listdir(folder)):
            if not filename.lower().endswith(valid_extensions):
                continue
            try:
                path = os.path.join(folder, filename)
                image = cv2.imread(path)
                if image is None:
                    print(f"Could not read image: {filename}")
                    continue
                    
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                faces = self.detector.detect_faces(rgb)
                
                if len(faces) == 0:
                    print(f"No faces found in: {filename}")
                    continue
                    
                face = max(faces, key=lambda r: r['box'][2] * r['box'][3])
                aligned = self.align_face(rgb, face)
                
                if aligned is None:
                    print(f"Alignment failed for: {filename}")
                    continue
                    
                # Resize to expected input size
                aligned = cv2.resize(aligned, (160, 160))
                
                embedding = self.embedder.embeddings([aligned])[0]
                
                # Verify embedding dimension
                if len(embedding) != self.EMBEDDING_DIM:
                    print(f"Error: Expected {self.EMBEDDING_DIM}D embedding, got {len(embedding)}D")
                    continue
                
                embeddings.append(embedding)
                print(f"Processed: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

        self.save_registration(user_id, name, dept, hashed_password, email, embeddings)

    def save_registration(self, user_id, name, dept, hashed_password, email, embeddings):
        if not embeddings:
            print("Registration failed - no valid embeddings created")
            return

        mean_embedding = np.mean(embeddings, axis=0)
        with open(self.embeddings_file, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([user_id] + mean_embedding.tolist())

        with open(self.user_details_file, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([user_id, name, dept, hashed_password, email])

        print(f"Successfully registered {name} with {len(embeddings)} samples")

    def mark_attendance_for_course(self, course_code, course_name, session):
        """Modified attendance marking that includes course information"""
           # Create directory for attendance records
        attendance_dir = "attendance_records"
        os.makedirs(attendance_dir, exist_ok=True)
        
        # Generate filename based on course and timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"attendance_{course_code}_{timestamp}.csv"
        filepath = os.path.join(attendance_dir, filename)
        
        # Initialize attendance dictionary
        attendance = {}
        
        # Initialize face recognition variables
        known_face_encodings = []
        known_student_ids = []
        
        # Load embeddings with dimension check
        if not os.path.exists(self.embeddings_file):
            print("No registered users found!")
            return

        emb_data = pd.read_csv(self.embeddings_file)
        if emb_data.empty:
            print("Error: Embeddings file is empty!")
            return

        known_student_ids = emb_data["UserID"].tolist()
        known_face_encodings = emb_data.drop("UserID", axis=1).values

        # Verify embedding dimensions
        if known_face_encodings.shape[1] != self.EMBEDDING_DIM:
            print(f"FATAL: Dimension mismatch! Expected {self.EMBEDDING_DIM}D, got {known_face_encodings.shape[1]}D")
            return
        embedding_norms = np.linalg.norm(known_face_encodings, axis=1)
        
        # Filter by enrolled students if set
        if self.student_ids:
            filtered_encodings = []
            filtered_student_ids = []
            for i, sid in enumerate(known_student_ids):
                if sid in self.student_ids:
                    filtered_encodings.append(known_face_encodings[i])
                    filtered_student_ids.append(sid)
            
            if filtered_encodings:
                known_face_encodings = filtered_encodings
                known_student_ids = filtered_student_ids
                print(f"Filtered to {len(known_face_encodings)} enrolled students")
            else:
                print(f"No enrolled students found in facial database for {course_code}")
                return
        
        if not os.path.exists(self.embeddings_file):
            print("No registered users found!")
            return

        # Load embeddings with dimension check
        emb_data = pd.read_csv(self.embeddings_file)
        if emb_data.empty:
            print("Error: Embeddings file is empty!")
            return
            
        user_ids = emb_data["UserID"].values
        embeddings = emb_data.drop("UserID", axis=1).values
        
        # Verify embedding dimensions
        if embeddings.shape[1] != self.EMBEDDING_DIM:
            print(f"FATAL: Dimension mismatch! Expected {self.EMBEDDING_DIM}D, got {embeddings.shape[1]}D")
            return

        # Precompute norms for efficiency
        embedding_norms = np.linalg.norm(embeddings, axis=1)
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not access camera!")
            return
            
        present = set()
        attendance = []
        last_update_time = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Camera error, skipping frame")
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = self.detector.detect_faces(rgb)

                for face in faces:
                    try:
                        # Skip small faces
                        if face['box'][2] < 100 or face['box'][3] < 100:
                            continue
                            
                        aligned = self.align_face(rgb, face)
                        if aligned is None or aligned.size == 0:
                            continue
                            
                        # Resize to model input size
                        aligned = cv2.resize(aligned, (160, 160))
                            
                        query_emb = self.embedder.embeddings([aligned])[0]
                        
                        # Verify embedding dimension
                        if len(query_emb) != self.EMBEDDING_DIM:
                            print(f"Runtime error: Expected {self.EMBEDDING_DIM}D, got {len(query_emb)}D")
                            continue

                        # Optimized cosine similarity
                        dot_product = np.dot(embeddings, query_emb)
                        query_norm = np.linalg.norm(query_emb)
                        similarities = dot_product / (embedding_norms * query_norm + 1e-8)
                        
                        best_match = np.argmax(similarities)
                        confidence = similarities[best_match]
                        x, y, w, h = face['box']

                        if confidence > self.face_threshold:
                            user_id = user_ids[best_match]
                            if user_id not in present:
                                # Throttle attendance recording
                                if time.time() - last_update_time > 1.5:
                                    present.add(user_id)
                                    details = self.get_user_details(user_id)
                                    attendance.append({
                                        "ID": user_id,
                                        "Name": details["Name"],
                                        "Status": 'Present',
                                        "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                        "Confidence": f"{confidence*100:.1f}%",
                                        "CourseCode": course_code,
                                        "CourseName": course_name,
                                        "Session": session
                                    })
                                    last_update_time = time.time()
                                label = f"{details["Name"]} ({user_id})"
                                color = (0, 255, 0)
                            else:
                                label = f"{details["Name"]} ({user_id}) (Already marked)"
                                color = (0, 165, 255)  # Orange
                        else:
                            label = "Unknown"
                            color = (0, 0, 255)

                        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                        cv2.putText(frame, label, (x+5, y-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        cv2.putText(frame, f"Conf: {confidence*100:.1f}%", 
                                    (x+5, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                    except Exception as e:
                        print(f"Face processing error: {str(e)}")
                        continue

                # Display instructions
                cv2.putText(frame, f"Course: {course_code} - {session}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, "Press ESC to exit", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow(f"Attendance: {course_name}", frame)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                    break
                    
        finally:
            cap.release()
            cv2.destroyAllWindows()

        if attendance:
            filename = f"attendance_{course_code}_{session}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            save_path = os.path.join(self.attendance_dir, filename)
            pd.DataFrame(attendance).to_csv(save_path, index=False)
            print(f"\nAttendance saved to {filename}")

    def get_user_details(self, user_id):
        try:
            with open(self.user_details_file, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["UserID"] == user_id:
                        return row
        except Exception as e:
            print(f"Error reading user details: {str(e)}")
            
        return {"Name": "Unknown", "Department": "N/A"}

    # ================== LECTURER MANAGEMENT ==================
    def add_lecturer(self):
        """Add a lecturer without face registration"""
        print("\n" + "="*20 + " ADD LECTURER " + "="*20)
        lecturer_id = input("Enter Lecturer ID: ").strip()
        name = input("Full Name: ").strip()
        dept = input("Department: ").strip()
        email = input("Email: ").strip()
        password = getpass.getpass("Set Password: ")
        hashed_password = self.hash_password(password)
        
        # Save to lecturers file
        with open(self.lecturers_file, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([lecturer_id, name, dept, email, hashed_password])
        
        print(f"Successfully added lecturer {name}")

    def view_lecturers(self):
        """Display all lecturers and their courses"""
        print("\n" + "="*20 + " LECTURER LIST " + "="*20)
        try:
            with open(self.lecturers_file, "r") as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader, 1):
                    print(f"{i}. {row['Name']} ({row['LecturerID']})")
                    print(f"   Department: {row['Department']}")
                    print(f"   Email: {row['Email']}")
                    
                    # Show assigned courses
                    assigned_courses = self.get_lecturer_courses(row['LecturerID'])
                    if assigned_courses:
                        print("   Assigned Courses:")
                        for course in assigned_courses:
                            print(f"     - {course['CourseName']} ({course['CourseCode']}) - {course['Session']}")
                    else:
                        print("   No courses assigned")
                    print("-" * 40)
        except Exception as e:
            print(f"Error reading lecturers: {str(e)}")

    # ================== COURSE MANAGEMENT ==================
    def create_course(self):
        """Create a new course"""
        print("\n" + "="*20 + " CREATE COURSE " + "="*20)
        course_code = input("Enter Course Code: ").strip()
        course_name = input("Enter Course Name: ").strip()
        session = input("Enter Session (e.g., Fall 2023): ").strip()
        
        # Save to courses file
        with open(self.courses_file, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([course_code, course_name, session, ""])  # Empty lecturer ID initially
        
        print(f"Successfully created course: {course_name} ({course_code})")

    def get_lecturer_courses(self, lecturer_id):
        """Get all courses assigned to a lecturer"""
        courses = []
        try:
            with open(self.courses_file, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["LecturerID"] == lecturer_id:
                        courses.append(row)
        except:
            pass
        return courses

    def assign_course_to_lecturer(self):
        """Assign a course to a lecturer"""
        print("\n" + "="*20 + " ASSIGN COURSE " + "="*20)
        
        # List available courses
        print("\nAvailable Courses:")
        courses = []
        try:
            with open(self.courses_file, "r") as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader, 1):
                    courses.append(row)
                    lecturer_name = "Unassigned"
                    if row["LecturerID"]:
                        lecturer = self.get_lecturer_details(row["LecturerID"])
                        lecturer_name = lecturer.get("Name", "Unknown") if lecturer else "Unknown"
                    print(f"{i}. {row['CourseName']} ({row['CourseCode']}) - {row['Session']} - Lecturer: {lecturer_name}")
        except Exception as e:
            print(f"Error reading courses: {str(e)}")
            return
        
        if not courses:
            print("No courses available!")
            return
            
        # Select course to assign
        course_index = input("\nEnter course number to assign: ").strip()
        try:
            course_index = int(course_index) - 1
            if course_index < 0 or course_index >= len(courses):
                print("Invalid course selection!")
                return
            selected_course = courses[course_index]
        except:
            print("Invalid input!")
            return
        
        # List lecturers
        lecturers = []
        try:
            with open(self.lecturers_file, "r") as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader, 1):
                    lecturers.append(row)
                    print(f"{i}. {row['Name']} ({row['LecturerID']})")
        except Exception as e:
            print(f"Error reading lecturers: {str(e)}")
            return
        
        if not lecturers:
            print("No lecturers available!")
            return
            
        # Select lecturer to assign
        lecturer_index = input("\nEnter lecturer number to assign: ").strip()
        try:
            lecturer_index = int(lecturer_index) - 1
            if lecturer_index < 0 or lecturer_index >= len(lecturers):
                print("Invalid lecturer selection!")
                return
            selected_lecturer = lecturers[lecturer_index]
        except:
            print("Invalid input!")
            return
        
        # Update the course assignment
        updated_courses = []
        try:
            with open(self.courses_file, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["CourseCode"] == selected_course["CourseCode"]:
                        row["LecturerID"] = selected_lecturer["LecturerID"]
                    updated_courses.append(row)
        except:
            pass
        
        # Save updated courses
        with open(self.courses_file, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["CourseCode", "CourseName", "Session", "LecturerID"])
            for course in updated_courses:
                writer.writerow([course["CourseCode"], course["CourseName"], course["Session"], course["LecturerID"]])
        
        print(f"\nSuccessfully assigned {selected_course['CourseName']} to {selected_lecturer['Name']}")

    def get_lecturer_details(self, lecturer_id):
        """Get lecturer details by ID"""
        try:
            with open(self.lecturers_file, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["LecturerID"] == lecturer_id:
                        return row
        except:
            pass
        return None

    def view_courses(self):
        """Display all courses and their assignments"""
        print("\n" + "="*20 + " COURSE LIST " + "="*20)
        try:
            with open(self.courses_file, "r") as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader, 1):
                    lecturer_name = "Unassigned"
                    if row["LecturerID"]:
                        lecturer = self.get_lecturer_details(row["LecturerID"])
                        lecturer_name = lecturer.get("Name", "Unknown") if lecturer else "Unknown"
                    print(f"{i}. {row['CourseName']} ({row['CourseCode']})")
                    print(f"   Session: {row['Session']}")
                    print(f"   Lecturer: {lecturer_name}")
                    print("-" * 40)
        except Exception as e:
            print(f"Error reading courses: {str(e)}")
    
    def view_students_by_department(self):
        """View students grouped by department"""
        students = pd.read_csv(self.user_details_file)

        if students.empty:
            print("No students found in the file.")
            return

        # Group students by department
        dept_groups = {}
        for _, student in students.iterrows():
            dept = student.get('Department', 'Undeclared')
            if dept not in dept_groups:
                dept_groups[dept] = []
            dept_groups[dept].append(student)

        # Print students by department
        print("\n" + "=" * 80)
        print("STUDENTS BY DEPARTMENT")
        print("=" * 80)

        for dept, student_list in dept_groups.items():
            print(f"\nDepartment: {dept}")
            print("-" * 40)
            for idx, student in enumerate(student_list, 1):
                print(f"{idx}. {student['Name']} {student['UserID']}")
            print(f"Total: {len(student_list)} students")

    def view_course_attendance_summary(self):
        """View attendance summary for a specific course"""
        courses = pd.read_csv(self.courses_file)

        if courses.empty:
            print("No courses found in the file.")
            return

        # List available courses
        print("\nAvailable Courses:")
        for idx, (_, course) in enumerate(courses.iterrows(), 1):
            print(f"{idx}. {course['CourseName']} ({course['CourseCode']}) - {course['Session']}")

        try:
            choice = int(input("\nSelect a course (number): "))
            if choice < 1 or choice > len(courses):
                raise IndexError
            selected_course = courses.iloc[choice - 1]  # Correct row selection
        except (ValueError, IndexError):
            print("Invalid selection!")
            return

        # Generate attendance report for selected course
        self.generate_course_attendance_report(selected_course)


    def generate_course_attendance_report(self, course):
        """Generate attendance report for a specific course"""
        
        self.current_course = course
        
        course_code = course["CourseCode"]
        session = course["Session"]
        
        # Find attendance files for this course
        attendance_files = []
        for filename in os.listdir(self.attendance_dir):
            if filename.startswith(f"attendance_{course_code}_") and filename.endswith(".csv"):
                # Extract date from filename
                date_str = filename.split('_')[2].split('.')[0]
                attendance_files.append((filename, date_str))
        
        if not attendance_files:
            print(f"No attendance records found for {course['CourseName']}!")
            return
        
        # Sort by date
        attendance_files.sort(key=lambda x: x[1])
        
        # Load all attendance records
        all_records = []
        unique_dates = set()
        for filename, date_str in attendance_files:
            try:
                file_path = os.path.join(self.attendance_dir, filename)
                df = pd.read_csv(file_path)
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
        
        # Get enrolled students
        self.current_course = course
        students = self.get_course_students()
        
        if not students:
            print("No students enrolled in this course!")
            return
        student_df = pd.DataFrame(students)
        
        # Create complete student-date matrix
        attendance_dates = sorted(unique_dates)
        student_df['key'] = 1
        dates_df = pd.DataFrame({'SessionDate': attendance_dates, 'key': 1})
        full_matrix = student_df.merge(dates_df, on='key').drop('key', axis=1)
        
        # Merge with attendance data
        merged_df = full_matrix.merge(
            combined_df[['ID', 'SessionDate', 'Status']],
            on=['ID', 'SessionDate'],
            how='left',
            suffixes=('', '_y')
        )
        
        # Clean status (Present/Absent)
        merged_df['Status'] = merged_df['Status'].fillna('Absent')
        merged_df = merged_df[['ID', 'Name', 'SessionDate', 'Status']]
        
        # Create pivot table
        pivot_df = merged_df.pivot_table(
            index=['ID', 'Name'],
            columns='SessionDate',
            values='Status',
            aggfunc='first',
            fill_value='Absent'
        ).reset_index()
        
        # Calculate statistics
        date_columns = [col for col in pivot_df.columns if col not in ['ID', 'Name']]
        total_sessions = len(date_columns)
        
        pivot_df['PresentCount'] = pivot_df[date_columns].apply(
            lambda row: (row == 'Present').sum(),
            axis=1
        )
        
        pivot_df['AttendancePercentage'] = (pivot_df['PresentCount'] / total_sessions) * 100
        
        # Display report
        print("\n" + "="*80)
        print(f"ATTENDANCE SUMMARY: {course['CourseName']} ({course_code}) - {session}")
        print("="*80)
        print(f"Total Sessions: {total_sessions}")
        print(f"Attendance Dates: {', '.join(attendance_dates)}")
        
        # Print summary statistics
        print("\nAttendance Summary:")
        print(f"{'Total Students:':<20} {len(pivot_df)}")
        print(f"{'Average Attendance:':<20} {pivot_df['AttendancePercentage'].mean():.2f}%")
        print(f"{'Highest Attendance:':<20} {pivot_df['AttendancePercentage'].max():.2f}%")
        print(f"{'Lowest Attendance:':<20} {pivot_df['AttendancePercentage'].min():.2f}%")
        
        # Show detailed report
        show_details = input("\nShow detailed student report? (y/n): ").lower()
        if show_details == 'y':
            print("\nDetailed Attendance Report:")
            print(pivot_df.to_string(index=False))
        
        # Save report option
        save_report = input("\nSave report to Excel? (y/n): ").lower()
        if save_report == 'y':
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_filename = f"attendance_summary_{course_code}_{session}_{timestamp}.xlsx"
            report_path = os.path.join(self.attendance_dir, report_filename)
            try:
                pivot_df.to_excel(report_path, index=False)
                print(f"Report saved to: {report_path}")
            except Exception as e:
                print(f"Error saving report: {str(e)}")
    
    def get_course_students(self):
        try:
            enroll_df = pd.read_csv(self.course_enrollments_file)
            print("Enrollment columns:", enroll_df.columns.tolist())  # Confirmed

            students_df = pd.read_csv(self.user_details_file)
            print("Students columns:", students_df.columns.tolist())  # Debug print

            enroll_df.columns = enroll_df.columns.str.strip()
            students_df.columns = students_df.columns.str.strip()

            course_code = self.current_course["CourseCode"]

            enrolled = enroll_df[enroll_df["CourseCode"] == course_code]

            merged = enrolled.merge(
                students_df,
                left_on="StudentID",
                right_on="UserID",
                how="left"
            )

            merged = merged.rename(columns={"StudentID": "ID"})
            merged = merged.dropna(subset=["Name"])

            return merged[["ID", "Name"]].to_dict(orient="records")
        except Exception as e:
            print(f"Error loading enrolled students: {str(e)}")
            return []




    def manage_courses(self):
        """Menu for course management"""
        while True:
            print("\n" + "="*20 + " COURSE MANAGEMENT " + "="*20)
            print("1. Create New Course")
            print("2. Assign Course to Lecturer")
            print("3. View All Courses")
            print("4. Back to Main Menu")
            choice = input("Select option: ").strip()

            if choice == "1":
                self.create_course()
            elif choice == "2":
                self.assign_course_to_lecturer()
            elif choice == "3":
                self.view_courses()
            elif choice == "4":
                break
            else:
                print("Invalid choice!")

    # ================== MAIN MENU SYSTEM ==================
    def manage_lecturers(self):
        """Menu for lecturer management"""
        while True:
            print("\n" + "="*20 + " LECTURER MANAGEMENT " + "="*20)
            print("1. Add New Lecturer")
            print("2. View All Lecturers")
            print("3. Back to Main Menu")
            choice = input("Select option: ").strip()

            if choice == "1":
                self.add_lecturer()
            elif choice == "2":
                self.view_lecturers()
            elif choice == "3":
                break
            else:
                print("Invalid choice!")

    def main_menu(self):
        while True:
            print("\n" + "="*20 + " MAIN MENU " + "="*20)
            print("1. Admin Login")
            print("2. Lecturer Login")
            print("3. Exit")
            choice = input("Select option: ").strip()

            if choice == "1":
                if self.authenticate_admin():
                    self.admin_menu()
            elif choice == "2":
                self.lecturer_login()
            elif choice == "3":
                print("Exiting system...")
                break
            else:
                print("Invalid choice!")

    def admin_menu(self):
        """Menu for admin functions"""
        while True:
            print("\n" + "="*20 + " ADMIN MENU " + "="*20)
            print("1. Register New User (Student)")
            print("2. Manage Lecturers")
            print("3. Manage Courses")
            print("4. View Students by Department")
            print("5. View Course Attendance Summary")
            print("6. Logout")
            choice = input("Select option: ").strip()

            if choice == "1":
                self.register_user()
            elif choice == "2":
                self.manage_lecturers()
            elif choice == "3":
                self.manage_courses()
            elif choice == "4":
                self.view_students_by_department()
            elif choice == "5":
                self.view_course_attendance_summary()
            elif choice == "6":
                print("Logging out...")
                break
            else:
                print("Invalid choice!")
            
    def lecturer_login(self):
        """Launch lecturer portal"""
        from lecturers_portal import LecturerPortal
        portal = LecturerPortal()
        if portal.lecturer_login():
            portal.select_course()
            portal.lecturer_menu()