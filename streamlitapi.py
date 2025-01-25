import pickle
import streamlit as st  
import numpy as np 

# Load the model
try:
    model_file = pickle.load(open(r'C:\Users\dalpa\datascienceinternship\model.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model file not found! Please check the file path.")
    st.stop()

def pred_output(user_input): 
    model_input = np.array(user_input)
    try:
        ypred = model_file.predict(model_input.reshape(1, -1))  # Reshape dynamically for safety
        return ypred[0]
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return None

def main(): 
    st.title("Titanic Survival Prediction")

    # Input Variables
    try:
        passenger_class = int(st.text_input("Enter the passenger class: (1/2/3)", value="1"))

        sex = st.radio("Select your sex:", options=["Male", "Female"])
        sex = 0 if sex == "Male" else 1

        age = float(st.text_input("Enter your age:", value="30"))

        sibsp = int(st.text_input("Enter the number of siblings:", value="0"))

        parch = int(st.text_input("Enter the number of parents/children aboard:", value="0"))

        fare = float(st.text_input("Enter the ticket fare:", value="50"))

        embarked = st.radio("Select port of embarkation:", options=["C", "Q", "S"])
        embarked = {"C": 1, "Q": 2, "S": 0}[embarked]

        # Button to predict
        if st.button('Predict'): 
            user_input = [passenger_class, sex, age, sibsp, parch, fare, embarked]
            make_prediction = pred_output(user_input)

            if make_prediction is not None:
                result = "Survived :)" if make_prediction == 1 else "Not Survived :("
                st.success(f"Prediction: {result}")
            else:
                st.error("Prediction failed. Please check your input values.")
    except Exception as e:
        st.error(f"Error in user input: {e}")

if __name__ == '__main__': 
    main()
