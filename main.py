from src.data_cleaning import *
from src.model_training import train_model
from src.model_evaluation import evaluate_model
import joblib
from sklearn.model_selection import train_test_split

def main():
    # Step 1: Load Data
    data = load_and_clean_data('data/raw_data.csv')
    
    # Step 2: Split Data
    X = data.drop('Price', axis=1)
    y = data['Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Step 3: Train the Model
    model = train_model(X_train, y_train)
    
    # Step 4: Evaluate the Model
    evaluate_model(model, X_test, y_test)
    
    # Step 5: Save the Model
    joblib.dump(model, 'src/trained_model.pkl')

if __name__ == "__main__":
    main()
