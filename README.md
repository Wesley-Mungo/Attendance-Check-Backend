# Facial-Recognition-based-attendence-system-for-Class-attendance
I created a smart attendance system, 
which will help in recognising the student's face and updating the databse with their Student Id and Student Name in that particular course. 
I used image processing to develop the automatic process.  
I used Face detection to identify a person’s face and face recognition to verify the identified person’s face with the one I have in our database.
Sure! Here's a  summary of My face attendance system:

1. The system uses facial recognition ( MTCNN and Facenet ) to automate student attendance.
2. Students’ face data is captured and stored during enrollment for later identification.
3. Lecturers can log in, view assigned courses, and initiate attendance sessions.
4. When attendance is initiated, the system recognizes student faces from a live camera feed.
5. Each recognized student is marked as "Present" in a CSV file tied to the course and session date.
6. Admins can assign courses to lecturers and manage student-course enrollments.
7. Attendance records are saved per course, session, and date in organized CSV files.
8. Email notifications can be sent to students using `` after each attendance session.
9. An attendance summary feature allows lecturers to view and export detailed reports.
10. The system provides statistics such as total sessions, individual attendance percentages, and overall averages.
