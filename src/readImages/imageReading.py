# this is the code for reading spreadsheet and images into a sqlite database with 3 tables linked by patient number
# Run the following pip to install libraries in the IDE (Integrated Development Environment)
# pip install --upgrade pip 
# pip install sqlite3
# pip install imageio
# pip install pandas
# pip install openpyxl

import os
import imageio
import numpy as np
import pandas as pd
import sqlite3

# Directory containing the images
DIR = "images"

# Load the spreadsheet
file_path = 'Data/PatientsData.xlsx'
spreadsheet = pd.read_excel(file_path)

# Define a function to save the image array to the database
def save_array_to_db(array, db_name, table_name, patient_number, pneumonia):
    # Convert the array to a binary format
    array_bytes = array.tobytes()

    # Connect to SQLite database (or create it if it doesn't exist)
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Create a table if it doesn't exist, including patient_number and pneumonia fields
    cursor.execute(f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_number INTEGER NOT NULL,
        pneumonia BOOLEAN NOT NULL,
        array_data BLOB,
        shape TEXT,
        FOREIGN KEY(patient_number) REFERENCES PatientInfo(patient_number)
    )
    """)
    print(f"Table {table_name} created (if it didn't exist).")

    # Insert the array data along with patient_number and pneumonia into the table
    cursor.execute(f"""
    INSERT INTO {table_name} (patient_number, pneumonia, array_data, shape) VALUES (?, ?, ?, ?)
    """, (patient_number, pneumonia, array_bytes, str(array.shape)))
    print(f"Data for patient {patient_number} inserted into table {table_name}.")

    # Commit the transaction and close the connection
    conn.commit()
    print("Transaction committed.")
    conn.close()
    print("Connection closed.")

# Define a function to save patient info to the database
def save_patient_info(db_name, patient_number, age, gender):
    # Connect to SQLite database (or create it if it doesn't exist)
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Create a table if it doesn't exist
    cursor.execute(f"""
    CREATE TABLE IF NOT EXISTS PatientInfo (
        patient_number INTEGER PRIMARY KEY,
        age INTEGER NOT NULL,
        gender TEXT NOT NULL
    )
    """)
    print("Table PatientInfo created (if it didn't exist).")

    # Insert the patient info into the table
    cursor.execute(f"""
    INSERT INTO PatientInfo (patient_number, age, gender) VALUES (?, ?, ?)
    """, (patient_number, age, gender))
    print(f"Patient info for patient {patient_number} inserted into table PatientInfo.")

    # Commit the transaction and close the connection
    conn.commit()
    print("Transaction committed.")
    conn.close()
    print("Connection closed.")

# Define a function to save patient test results to the database
def save_patient_test_results(db_name, patient_number, hgb, platelets, rbc, hematocrit, wbc):
    # Connect to SQLite database (or create it if it doesn't exist)
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Create a table if it doesn't exist
    cursor.execute(f"""
    CREATE TABLE IF NOT EXISTS PatientTestResult (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_number INTEGER NOT NULL,
        hgb REAL NOT NULL,
        platelets REAL NOT NULL,
        rbc REAL NOT NULL,
        hematocrit REAL NOT NULL,
        wbc REAL NOT NULL,
        FOREIGN KEY(patient_number) REFERENCES PatientInfo(patient_number)
    )
    """)
    print("Table PatientTestResult created (if it didn't exist).")

    # Insert the patient test results into the table
    cursor.execute(f"""
    INSERT INTO PatientTestResult (patient_number, hgb, platelets, rbc, hematocrit, wbc) VALUES (?, ?, ?, ?, ?, ?)
    """, (patient_number, hgb, platelets, rbc, hematocrit, wbc))
    print(f"Test results for patient {patient_number} inserted into table PatientTestResult.")

    # Commit the transaction and close the connection
    conn.commit()
    print("Transaction committed.")
    conn.close()
    print("Connection closed.")

# Iterate over each record in the spreadsheet
for index, row in spreadsheet.iterrows():
    patient_number = row['Patient No']
    age = row['Patient Age']
    gender = row['Patient Gender']
    xray_file = row['Patient X-Ray File']
    pneumonia = row['Pneumonia']
    hgb = row['HGB, g/dL']
    platelets = row['Platelets, k/uL']
    rbc = row['RBC, M/uL']
    hematocrit = row['Hematocrit']
    wbc = row['WBC, M/uL']

    # Load the X-ray image
    image_path = os.path.join(DIR, xray_file)
    if os.path.exists(image_path):
        xray_image = imageio.v3.imread(image_path)
        image_array = np.array(xray_image)
        
        # Save the data to the database
        save_patient_info('Data/xRayImages.sqlite', patient_number, age, gender)
        save_patient_test_results('Data/xRayImages.sqlite', patient_number, hgb, platelets, rbc, hematocrit, wbc)
        save_array_to_db(image_array, 'Data/xRayImages.sqlite', 'PatientImages', patient_number, pneumonia)
    else:
        print(f"Image file {xray_file} not found for patient {patient_number}.")
