# Smart Attendance Analyzer

Smart Attendance Analyzer is a small Flask web application that uses attendance data mining and a Random Forest classifier to predict whether a student is **At Risk** or **Regular**. The app also generates subject-wise bar graphs and pie charts to make attendance patterns easy to compare.

## Features

- Six-subject attendance input form
- Present and absent counts for each subject
- Automatic attendance percentage calculation
- Total attended classes and total absentee count
- Random Forest based classification
- Weak subject detection using a 75% threshold
- Dataset-driven bar charts and pie charts

## Subjects

The project works with these six subjects only:

- DBMS
- DM
- EAD
- EEA
- ES
- PQT

## How It Works

1. The user enters present and absent values for each subject.
2. The backend calculates subject-wise attendance percentage:

   `attendance % = present / (present + absent) * 100`

3. It computes:

   - total attended classes
   - total absentee count

4. The model predicts the risk level using the attendance features.
5. Any subject below **75%** is marked as a weak subject.
6. The app displays the prediction and generated graphs.

## Dataset

The CSV file is used for training the model and generating the graphs.

Current schema:

- `Student_ID`
- `DBMS_Present`, `DBMS_Absent`
- `DM_Present`, `DM_Absent`
- `EAD_Present`, `EAD_Absent`
- `EEA_Present`, `EEA_Absent`
- `ES_Present`, `ES_Absent`
- `PQT_Present`, `PQT_Absent`
- `Attended_Classes`
- `Absentee_Count`
- `Risk_Label`

The dataset currently contains 1500 sample records.

## Graphs

The result page shows four graphs generated from the dataset:

- Subject-wise attendance bar chart
- Subject-wise absentees bar chart
- Attendance share pie chart
- Absentee share pie chart

The charts are saved inside the `static/` folder.

## Project Structure

```text
app.py
dataset.csv
static/
  style.css
  attendance.png
  absent.png
  attendance_pie.png
  absent_pie.png
templates/
  index.html
  result.html
```

## Installation

Install the required Python packages:

```bash
pip install flask pandas scikit-learn matplotlib
```

## Run the App

Start the Flask application:

```bash
python app.py
```

Then open the local server in your browser.

## Notes

- The app uses a 75% attendance threshold to identify weak subjects.
- The final prediction is aligned with the attendance rule shown on the result page.
- The charts are based on the training dataset, not the single user input.

## Use Case

This project is useful for a data mining or machine learning mini project because it demonstrates:

- preprocessing
- feature engineering
- supervised classification
- visual analysis with graphs
