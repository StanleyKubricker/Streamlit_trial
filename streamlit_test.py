# Import Streamlit
import streamlit as st

# Define the function
def reverse_string(input_string):
    return input_string[::-1]

# Title for the app
st.title('String Reversal App')

# Take string input from the user
input_string = st.text_input('Enter a string')

# Reverse the string
result = reverse_string(input_string)

# Display the result
st.write(f'The reverse of "{input_string}" is "{result}"')