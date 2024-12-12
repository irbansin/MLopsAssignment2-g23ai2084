import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the Iris dataset (or any dataset of your choice)
data = load_iris()
X = data.data
y = data.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define hyperparameters for both models
decision_tree_params = {'max_depth': 5, 'min_samples_split': 2}
random_forest_params = {'n_estimators': 100, 'max_depth': 10}

# Start an MLflow experiment
mlflow.set_experiment("Decision_Tree_Random_Forest_Experiment")

# Function to train and log Decision Tree model
def train_decision_tree(params):
    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_params(params)

        # Create and train the Decision Tree model
        model = DecisionTreeClassifier(max_depth=params['max_depth'], min_samples_split=params['min_samples_split'])
        model.fit(X_train, y_train)

        # Make predictions and evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)

        # Log model
        mlflow.sklearn.log_model(model, "decision_tree_model")
        
        print(f"Decision Tree Accuracy: {accuracy}")

# Function to train and log Random Forest model
def train_random_forest(params):
    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_params(params)

        # Create and train the Random Forest model
        model = RandomForestClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'])
        model.fit(X_train, y_train)

        # Make predictions and evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)

        # Log model
        mlflow.sklearn.log_model(model, "random_forest_model")
        
        print(f"Random Forest Accuracy: {accuracy}")

# Run both models and log their results
train_decision_tree(decision_tree_params)
train_random_forest(random_forest_params)
