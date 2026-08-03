
import pandas as pd
import os
import gradio as gr
import numpy as np
# Import required tools
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print(os.listdir('/'))


# Automatically find the uploaded CSV file
#csv_files = glob.glob("*.csv") + glob.glob("/content/*.csv") + glob.glob("/*.csv")

#print("CSV files found:", csv_files)

df = pd.read_csv("placement_predict_50k Dataset.csv")

print("Dataset loaded successfully!")
print("Rows and Columns:", df.shape)

df.head()

#csv_files = glob.glob('/*.csv')
#print(csv_files)
print("ALL COLUMNS IN THE DATASET:\n")

for i, column in enumerate(df.columns, start=1):
    print(i, ":", column)
print("Placement Status values:")
print(df["PlacementStatus"].value_counts())

print("\nMissing Placement Status values:")
print(df["PlacementStatus"].isnull().sum())

print("\nSample values:")
print(df["PlacementStatus"].head(10))
columns_to_remove = [
    "StudentID",
    "PlacementStatus",
    "IsAnomaly",
    "Salary Package"
]

# X = information used to make prediction
X = df.drop(columns=columns_to_remove)

# y = answer we want the model to learn
y = df["PlacementStatus"]

print("Input features:", X.shape[1])
print("Target:", "PlacementStatus")

print("\nFeatures being used:")
for i, col in enumerate(X.columns, start=1):
    print(i, ":", col)

print("\nTarget distribution:")
print(y.value_counts())
print("DATA TYPES:\n")
print(X.dtypes)

print("\n-------------------------")
print("MISSING VALUES:\n")
print(X.isnull().sum())
# 1. Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

print("Categorical columns:", len(categorical_cols))
print("Numerical columns:", len(numerical_cols))

# 2. Numeric preprocessing:
# Fill missing numeric values using median
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

# 3. Categorical preprocessing:
# Fill missing categories if any and convert text to numbers
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# 4. Combine preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)

# 5. Create Random Forest model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

# 6. Combine preprocessing + model
placement_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", model)
])

# 7. Split dataset: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# 8. Train the model
print("\nTraining PlacementPredict...")

placement_model.fit(X_train, y_train)

print("Training completed!")

# 9. Make predictions on test data
y_pred = placement_model.predict(X_test)

# 10. Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nPLACEMENTPREDICT RESULTS")
print("-------------------------")
print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Not Placed", "Placed"]
))

def predict_placement(
    cgpa,
    attendance,
    internships,
    projects,
    certifications,
    aptitude,
    softskills,
    coding,
    mockinterview
):

    # Start with typical/default values from the training data
    student = {}

    for col in X.columns:

        if X[col].dtype == "object":
            # Most common category
            student[col] = X[col].mode()[0]

        else:
            # Median numeric value
            student[col] = X[col].median()

    # Replace defaults with values entered by the student
    student["CGPA"] = cgpa
    student["AttendancePercent"] = attendance
    student["Internships"] = internships
    student["Projects"] = projects
    student["Certifications"] = certifications
    student["AptitudeTestScore"] = aptitude
    student["SoftSkillsRating"] = softskills
    student["CodingTestScore"] = coding
    student["MockInterviewScore"] = mockinterview

    # Keep CGPA Tier consistent with CGPA
    # Use the nearest CGPA value in the original dataset
    nearest_index = (df["CGPA"] - cgpa).abs().idxmin()
    student["CGPA_Tier"] = df.loc[nearest_index, "CGPA_Tier"]

    # Convert to DataFrame
    student_df = pd.DataFrame([student])

    # Ensure same column order used during training
    student_df = student_df[X.columns]

    # Prediction
    prediction = placement_model.predict(student_df)[0]

    # Probability
    probabilities = placement_model.predict_proba(student_df)[0]

    placed_probability = probabilities[1] * 100
    not_placed_probability = probabilities[0] * 100

    if prediction == 1:

        result = f"""
LIKELY TO BE PLACED

Placement Probability: {placed_probability:.2f}%

Not-Placed Probability: {not_placed_probability:.2f}%
"""

    else:

        result = f"""
NOT LIKELY TO BE PLACED

Placement Probability: {placed_probability:.2f}%

Not-Placed Probability: {not_placed_probability:.2f}%
"""

    return result


# -------------------------------------------------
# CREATE GRADIO INTERFACE
# -------------------------------------------------

demo = gr.Interface(

    fn=predict_placement,

    inputs=[

        gr.Slider(
            minimum=0,
            maximum=10,
            value=8,
            step=0.1,
            label="CGPA"
        ),

        gr.Slider(
            minimum=0,
            maximum=100,
            value=80,
            step=1,
            label="Attendance Percentage"
        ),

        gr.Slider(
            minimum=0,
            maximum=10,
            value=1,
            step=1,
            label="Number of Internships"
        ),

        gr.Slider(
            minimum=0,
            maximum=10,
            value=2,
            step=1,
            label="Number of Projects"
        ),

        gr.Slider(
            minimum=0,
            maximum=10,
            value=2,
            step=1,
            label="Number of Certifications"
        ),

        gr.Slider(
            minimum=0,
            maximum=100,
            value=70,
            step=1,
            label="Aptitude Test Score"
        ),

        gr.Slider(
            minimum=0,
            maximum=10,
            value=7,
            step=0.1,
            label="Soft Skills Rating"
        ),

        gr.Slider(
            minimum=0,
            maximum=100,
            value=70,
            step=1,
            label="Coding Test Score"
        ),

        gr.Slider(
            minimum=0,
            maximum=100,
            value=70,
            step=1,
            label="Mock Interview Score"
        )
    ],

    outputs=gr.Textbox(
        label="Placement Prediction",
        lines=6
    ),

    title="🎓 PlacementPredict",

    description="""
Enter the student's details below and click Submit
to predict the likelihood of placement.
"""
)

# Launch the application
demo.launch(share=True)
