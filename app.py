"""
Student Performance Prediction System
All-in-one Flask application with model training functionality

This combined approach eliminates path issues by keeping everything in one file.
"""
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
import pickle
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.base import BaseEstimator, ClassifierMixin

app = Flask(__name__)

# Create necessary directories
os.makedirs('models', exist_ok=True)
os.makedirs('static', exist_ok=True)

# Define paths with absolute references to avoid any path issues
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(CURRENT_DIR, "data", "student_performance.csv")
MODELS_DIR = os.path.join(CURRENT_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "model.pkl")
FEATURE_NAMES_PATH = os.path.join(MODELS_DIR, "feature_names.pkl")
GRADE_MAPPING_PATH = os.path.join(MODELS_DIR, "grade_mapping.pkl")
FEATURE_WEIGHTS_PATH = os.path.join(MODELS_DIR, "feature_weights.pkl")

# Define custom importance weights
CUSTOM_WEIGHTS = {
    'Weekly_Study_Hours': 0.25,  # Highest importance
    'Attendance': 0.20,
    'Listening_in_Class': 0.15,
    'Project_work': 0.12,
    'Notes': 0.10,
    'Reading': 0.08,
    'Scholarship': 0.03,
    'Student_Age': 0.02,
    'High_School_Type': 0.02,
    'Sex': 0.01,
    'Sports_activity': 0.01,
    'Additional_Work': 0.01,
    'Transportation': 0.01
}

# Variables to store model and metadata
model = None
FEATURE_NAMES = []
GRADE_MAPPING = {}

# Custom wrapper class for feature importance
class CustomImportanceRandomForest(BaseEstimator, ClassifierMixin):
    def __init__(self, base_model=None, custom_importance=None):
        if base_model is None:
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                random_state=42
            )
        else:
            self.model = base_model
            
        self._custom_importance = custom_importance
        self._feature_names = None
        
    def fit(self, X, y):
        if hasattr(X, 'columns'):
            self._feature_names = X.columns.tolist()
        self.model.fit(X, y)
        return self
        
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    @property
    def feature_importances_(self):
        return self._custom_importance if self._custom_importance is not None else self.model.feature_importances_
    
    @property
    def classes_(self):
        return self.model.classes_
        
    @property
    def feature_names_in_(self):
        if hasattr(self.model, 'feature_names_in_'):
            return self.model.feature_names_in_
        return np.array(self._feature_names) if self._feature_names else None

def convert_age_range(age_str):
    """Convert age range strings like '19-22' to their average value"""
    try:
        if isinstance(age_str, (int, float)):
            return age_str
        if '-' in str(age_str):
            parts = str(age_str).split('-')
            nums = [int(p.strip()) for p in parts]
            return sum(nums) / len(nums)
        return float(age_str)
    except:
        return 20.0  # Default age

def train_model():
    """Train the model and save all necessary files"""
    global model, FEATURE_NAMES, GRADE_MAPPING

    print("\n===== Training Student Performance Prediction Model =====\n")

    # Check if dataset exists
    if not os.path.exists(DATA_PATH):
        print(f"Dataset not found at {DATA_PATH}")
        print("Please ensure the dataset is in the data directory")
        # Create mock data if needed for demonstration
        create_mock_data()
        
    # Load dataset
    try:
        df = pd.read_csv(DATA_PATH)
        print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Creating mock dataset instead...")
        df = create_mock_data()

    # Drop Student_ID if present
    if 'Student_ID' in df.columns:
        df = df.drop('Student_ID', axis=1)
        print("Dropped Student_ID column as it's not a predictive feature")
    
    # Convert Student_Age column from ranges to numeric values
    if 'Student_Age' in df.columns:
        print("Converting Student_Age values...")
        df['Student_Age'] = df['Student_Age'].apply(convert_age_range)
    
    # Identify and encode target variable
    target_column = 'Grade'
    if target_column not in df.columns:
        if 'Performance' in df.columns:
            target_column = 'Performance'
        else:
            print(f"Target column not found. Available columns: {df.columns.tolist()}")
            # Add a mock target column
            df['Grade'] = np.random.randint(0, 5, size=len(df))
            target_column = 'Grade'
    
    # Save original grade values before encoding
    original_grades = sorted(df[target_column].unique())
    print(f"Original grades found in dataset: {original_grades}")
    
    # Encode the target variable
    grade_encoder = LabelEncoder()
    encoded_grades = grade_encoder.fit_transform(df[target_column].astype(str))
    
    # Create mapping between encoded and original grades
    GRADE_MAPPING = dict(zip(grade_encoder.transform([str(g) for g in original_grades]), original_grades))
    print(f"Grade mapping: {GRADE_MAPPING}")
    
    # Save the grade mapping
    with open(GRADE_MAPPING_PATH, 'wb') as f:
        pickle.dump(GRADE_MAPPING, f)
    print(f"Grade mapping saved to {GRADE_MAPPING_PATH}")
    
    # Replace the target column with encoded values
    df[target_column] = encoded_grades
    
    # Encode all categorical features
    for col in df.columns:
        if col != target_column and df[col].dtype == 'object':
            print(f"Encoding categorical column: {col}")
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    
    # Split features and target
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    # Save feature names
    FEATURE_NAMES = X.columns.tolist()
    print(f"Features used for training: {FEATURE_NAMES}")
    
    # Save feature names to file
    with open(FEATURE_NAMES_PATH, 'wb') as f:
        pickle.dump(FEATURE_NAMES, f)
    print(f"Feature names saved to {FEATURE_NAMES_PATH}")
    
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    # Train the base model
    print("\nTraining Random Forest model...")
    base_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        random_state=42
    )
    base_model.fit(X_train, y_train)
    
    # Create custom importance array
    custom_importance = np.zeros(len(FEATURE_NAMES))
    
    # Normalize the weights to sum to 1
    total_weight = sum(CUSTOM_WEIGHTS.values())
    normalized_weights = {k: v/total_weight for k, v in CUSTOM_WEIGHTS.items()}
    
    # Add the custom weights in the correct order
    for i, feature in enumerate(FEATURE_NAMES):
        if feature in normalized_weights:
            custom_importance[i] = normalized_weights[feature]
            print(f"Setting {feature} importance to {normalized_weights[feature]:.4f}")
        else:
            custom_importance[i] = 0.001
            print(f"Setting {feature} importance to default 0.001")
    
    # Create the model with custom feature importance
    model = CustomImportanceRandomForest(base_model=base_model, custom_importance=custom_importance)
    
    # Save the model
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {MODEL_PATH}")
    
    # Save feature weights
    with open(FEATURE_WEIGHTS_PATH, 'wb') as f:
        pickle.dump(normalized_weights, f)
    print(f"Feature weights saved to {FEATURE_WEIGHTS_PATH}")
    
    # Generate feature importance visualization
    print("\nGenerating feature importance visualization...")
    create_feature_importance_plot(FEATURE_NAMES, custom_importance)
    
    print("\n===== Model training completed =====")
    return model

def create_mock_data():
    """Create mock data if dataset is missing"""
    print("Creating mock student performance dataset")
    
    # Number of samples
    n_samples = 150
    
    # Create mock features
    data = {
        'Student_Age': np.random.choice([18, 19, 20, 21, 22], size=n_samples),
        'Sex': np.random.randint(0, 2, size=n_samples),
        'High_School_Type': np.random.randint(0, 2, size=n_samples),
        'Scholarship': np.random.randint(0, 2, size=n_samples),
        'Additional_Work': np.random.randint(0, 2, size=n_samples),
        'Sports_activity': np.random.randint(0, 2, size=n_samples),
        'Transportation': np.random.randint(0, 3, size=n_samples),
        'Weekly_Study_Hours': np.random.randint(1, 40, size=n_samples),
        'Attendance': np.random.randint(50, 101, size=n_samples),
        'Reading': np.random.randint(0, 11, size=n_samples),
        'Notes': np.random.randint(0, 11, size=n_samples),
        'Listening_in_Class': np.random.randint(0, 11, size=n_samples),
        'Project_work': np.random.randint(0, 11, size=n_samples),
    }
    
    # Create target variable (grades: AA, BA, BB, CB, CC, DC, DD, Fail)
    grades = ['AA', 'BA', 'BB', 'CB', 'CC', 'DC', 'DD', 'Fail']
    data['Grade'] = np.random.choice(grades, size=n_samples)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Ensure data directory exists
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    
    # Save to CSV
    df.to_csv(DATA_PATH, index=False)
    print(f"Mock dataset saved to {DATA_PATH}")
    
    return df

def create_feature_importance_plot(feature_names, feature_importance):
    """Create and save a feature importance visualization"""
    # Sort features by importance
    idx = np.argsort(feature_importance)[::-1]
    sorted_names = [feature_names[i].replace('_', ' ').title() for i in idx]
    sorted_importance = feature_importance[idx]
    
    # Create plot
    plt.figure(figsize=(12, 8))
    colors = sns.color_palette("viridis", len(feature_names))
    y_pos = np.arange(len(feature_names))
    
    plt.barh(y_pos, sorted_importance, align='center', color=colors)
    plt.yticks(y_pos, sorted_names)
    plt.xlabel('Importance Score')
    plt.title('Feature Importance for Student Performance')
    
    # Add value labels
    for i, v in enumerate(sorted_importance):
        plt.text(v + 0.01, i, f"{v:.3f}", va='center')
    
    plt.tight_layout()
    
    # Save to static folder
    static_path = os.path.join(CURRENT_DIR, "static", "feature_importance.png")
    plt.savefig(static_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Feature importance graph saved to {static_path}")

def load_model_files():
    """Load model and related files"""
    global model, FEATURE_NAMES, GRADE_MAPPING
    
    try:
        # Check if model exists, if not, train it
        if not os.path.exists(MODEL_PATH):
            print("Model not found. Training now...")
            train_model()
        else:
            # Load model
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            print("Model loaded successfully")
            
            # Load feature names
            with open(FEATURE_NAMES_PATH, 'rb') as f:
                FEATURE_NAMES = pickle.load(f)
            print(f"Feature names loaded: {FEATURE_NAMES}")
            
            # Load grade mapping
            with open(GRADE_MAPPING_PATH, 'rb') as f:
                GRADE_MAPPING = pickle.load(f)
            print(f"Grade mapping loaded")
            
        return True
    except Exception as e:
        import traceback
        print(f"Error loading model files: {e}")
        print(traceback.format_exc())
        return False

def plot_feature_importance():
    """Get feature importance plot from saved model"""
    if not hasattr(model, 'feature_importances_'):
        return None
    
    # Get features and importance
    feature_importance = dict(zip(FEATURE_NAMES, model.feature_importances_))
    
    # Convert to DataFrame and sort
    feature_df = pd.DataFrame({
        'Feature': [name.replace('_', ' ').title() for name in feature_importance.keys()],
        'Importance': list(feature_importance.values())
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("viridis", len(feature_df))
    
    ax = sns.barplot(x='Importance', y='Feature', data=feature_df, palette=colors)
    
    # Add value annotations
    for i, v in enumerate(feature_df['Importance']):
        ax.text(v + 0.01, i, f"{v:.3f}", va='center')
    
    plt.title('Feature Importance for Student Performance')
    plt.tight_layout()
    
    # Convert to base64 for web display
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', dpi=300)
    img_buf.seek(0)
    img_data = base64.b64encode(img_buf.getvalue()).decode()
    plt.close()
    
    return img_data

def generate_study_plan(prediction):
    """Generate a personalized study plan based on prediction"""
    # Convert prediction to numeric scale if needed
    if isinstance(prediction, str):
        try:
            numeric_prediction = float(prediction)
        except:
            # If it's a letter grade like AA, BA, etc.
            letter_grade_map = {
                'AA': 5, 'BA': 4.5, 'BB': 4, 'CB': 3.5, 
                'CC': 3, 'DC': 2.5, 'DD': 2, 'Fail': 1
            }
            numeric_prediction = letter_grade_map.get(prediction, 3)
    else:
        numeric_prediction = float(prediction)
    
    # Scale numeric prediction to 1-5 scale if needed
    if numeric_prediction > 5:
        numeric_prediction = numeric_prediction / 20  # Assuming 100-point scale
    
    if numeric_prediction >= 4:
        category = "High Performer"
        focus_areas = ["Advanced concepts", "Competitive exam preparation"]
        recommended_hours = 20
    elif numeric_prediction >= 3:
        category = "Above Average"
        focus_areas = ["Concept strengthening", "Problem-solving techniques"]
        recommended_hours = 15
    elif numeric_prediction >= 2:
        category = "Average Performer"
        focus_areas = ["Regular practice", "Concept clarity"]
        recommended_hours = 12
    else:
        category = "Needs Improvement"
        focus_areas = ["Foundation concepts", "Daily structured practice"]
        recommended_hours = 10
    
    # Create weekly plan
    weekly_plan = {
        "Monday": f"Core concepts review - {max(1, recommended_hours//5)} hours",
        "Tuesday": f"Problem solving practice - {max(1, recommended_hours//5)} hours",
        "Wednesday": f"Review weak areas - {max(1, recommended_hours//5 + 1)} hours",
        "Thursday": f"Practice tests - {max(1, recommended_hours//5)} hours",
        "Friday": f"Group study/Project work - {max(1, recommended_hours//5)} hours",
        "Weekend": "Revision and rest"
    }
    
    study_plan = {
        "category": category,
        "focus_areas": focus_areas,
        "recommended_hours": recommended_hours,
        "weekly_plan": weekly_plan
    }
    
    return study_plan

# Initialize model function (replaces before_first_request)
def init_model():
    global model, FEATURE_NAMES, GRADE_MAPPING
    success = load_model_files()
    if not success:
        print("Failed to load model files. Training a new model...")
        train_model()

# Route for manual initialization
@app.route('/initialize')
def initialize():
    """Route to manually initialize the model"""
    init_model()
    return "Model initialized successfully! <a href='/'>Go back to prediction form</a>"

@app.route('/')
def home():
    """Home page with the input form"""
    global model, FEATURE_NAMES
    
    # If model or feature names not loaded, try loading them
    if model is None or not FEATURE_NAMES:
        success = load_model_files()
        if not success:
            # Train the model if loading fails
            train_model()
        
        # If still not loaded, return error
        if model is None or not FEATURE_NAMES:
            return "Error: Could not load or train model. Please check the logs."
    
    return render_template('index.html', feature_names=FEATURE_NAMES)

@app.route('/train', methods=['GET'])
def train():
    """Route to manually train/retrain the model"""
    train_model()
    return "Model trained successfully! <a href='/'>Go back to prediction form</a>"

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    global model, FEATURE_NAMES, GRADE_MAPPING
    
    # If model or feature names not loaded, try loading them
    if model is None or not FEATURE_NAMES:
        success = load_model_files()
        if not success:
            return jsonify({'error': 'Model not loaded. Please train the model first.'})
    
    try:
        # Get form data
        form_data = {}
        for field in request.form:
            if field != 'name':  # Skip name field
                try:
                    form_data[field] = float(request.form[field])
                except ValueError:
                    form_data[field] = request.form[field]
        
        # Create input data with the expected features
        input_data = {}
        for feature in FEATURE_NAMES:
            if feature in form_data:
                input_data[feature] = form_data[feature]
            else:
                print(f"Warning: Feature '{feature}' not found in form data. Using default.")
                input_data[feature] = 0  # Default value
        
        # Create DataFrame with correct feature order
        input_df = pd.DataFrame([input_data])
        input_df = input_df[FEATURE_NAMES]  # Ensure correct order
        
        # Make prediction
        encoded_prediction = model.predict(input_df)[0]
        
        # Convert encoded prediction to original grade
        original_grade = GRADE_MAPPING.get(encoded_prediction, str(encoded_prediction))
        
        # Get prediction confidence if available
        try:
            probabilities = model.predict_proba(input_df)[0]
            confidence = float(probabilities.max()) * 100
        except Exception as e:
            confidence = None
        
        # Get feature importance plot
        importance_img = plot_feature_importance()
        
        # Generate study plan
        study_plan = generate_study_plan(original_grade)
        
        # Render results
        return render_template('results.html',
                              name=request.form.get('name', 'Student'),
                              prediction=original_grade,
                              confidence=confidence,
                              importance_img=importance_img,
                              study_plan=study_plan)
    
    except Exception as e:
        import traceback
        print(f"Error during prediction: {e}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Initialize the model before running the app
    init_model()
    app.run(debug=True, host='0.0.0.0', port=5000)
