import sqlite3

## Connect to SQLite
connection=sqlite3.connect("student.db")

# Create a cursor object to insert record, create table
cursor=connection.cursor()

## create the table
table_info="""
Create table STUDENT(NAME VARCHAR(25), CLASS VARCHAR(25), SECTION VARCHAR(25),MARKS INT);

"""

cursor.execute(table_info)

## Insert Some more records

cursor.execute('''Insert Into STUDENT values('Ankit','CSE','A',90)''')
cursor.execute('''Insert Into STUDENT values('Krish','Data Science','A',100)''')
cursor.execute('''Insert Into STUDENT values('Tommy','CSE','B',86)''')
cursor.execute('''Insert Into STUDENT values('Anurag','Gate','B',35)''')
cursor.execute('''Insert Into STUDENT values('Ashish','DevOps','A',26)''')
cursor.execute('''Insert Into STUDENT values('Chandan','Gate','A',99)''')

# Display all the records
print("The inserted records are")
data=cursor.execute('''Select * from Student''')
for row in data:
    print(row)

## Commit your changes in the database
connection.commit()
connection.close()