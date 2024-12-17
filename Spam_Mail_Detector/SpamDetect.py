import streamlit as st
import pickle

# Load the trained model and vectorizer
model = pickle.load(open('spam.pkl', 'rb'))
cv = pickle.load(open('vectorizer.pkl', 'rb'))

# Main function for Streamlit app
def main():
    st.title('Spam E-mail Detector')
    st.subheader('Classification')

    # User input
    user_input = st.text_area('Input an email to detect spam')

    # Button for detection
    if st.button('Detect'):
        if user_input:
            # Vectorize input and predict
            data = [user_input]
            vec = cv.transform(data).toarray()
            res = model.predict(vec)
            if res[0] == 0:
                st.success('Not a SPAM')
            else:
                st.error('SPAM')
        else:
            st.warning('Please enter an email to classify.')

# Run the app
if __name__ == "__main__":
    main()
