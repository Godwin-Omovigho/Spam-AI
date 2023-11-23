import streamlit as st
import tensorflow as tf
import pandas as pd






loaded_model = tf.keras.models.load_model('Email CSV')


def show_predict_page():

    # Custom CSS for styling
    custom_css = """
    <style>

      .title {
              text-align: center;
              color: white;
              background-color: #00CED1; /* Background color is blue */
              padding: 10px; /* Add some padding for spacing */
          }
        .header {
            color: white;
            background-color: #00CED1; /* Background color is blue */
            padding: 5px 5px; /* Add some padding for spacing */
        }
        .stText {
            color: black; /* Change text color to black */
        }
        .header-text {
            color: white; /* Change text color to white for specific headers */
        }
    </style>
    """
    
    # Apply the custom CSS
    st.markdown(custom_css, unsafe_allow_html=True)

    # Streamlit app
    st.markdown("<h1 class='title'>Spam Email AI</h1>", unsafe_allow_html=True)
    
    
    # Description of your computer vision model
    model_description = "This model examines text messages to find and highlight those that show signs often linked to spam or unwanted messages."
    
    
    st.markdown("\n\n\n")
    st.markdown("<h2 class='header'> Model Description</h2>", unsafe_allow_html=True)
    st.write(model_description)

    st.markdown("<p style='font-size: 25px; font-weight: bold; color: red;'> Note: How well the model works depends on the data it learned from and how many examples there are for each group in the training data.</p>", unsafe_allow_html=True)

    # Create a textbox
    user_input = st.text_input("Enter text here:","")

    
    ok =st.button('Check message')
    #create a dataframe

    if ok:

      text=pd.DataFrame({'text':user_input},index=[1])

      test_sentences = text['text'].to_numpy()
      pred_prob = tf.squeeze(loaded_model.predict([test_sentences]))
      pred = tf.round(pred_prob)
      preds=pred_prob.numpy()

      # Display the model result
      if pred_prob >=0.9:
        # return st.subheader("Predicted class: Spam Message"), 
        return st.write("Predicted class:", "Spam Message",",","Prediction_Probability:", preds)

      else:
        # return st.subheader("Predicted class: Not Spam")
        return st.write("Predicted class:", "Not Spam",",","Prediction_Probability:", preds)
        
