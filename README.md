
# Construction Submittal Chat Bot

Welcome to the Construction Submittal Chat Bot project! This chat bot is designed on top of the ChatGPT model to facilitate conversations about uploaded construction submittals. The bot provides a user-friendly interface to discuss and interact with the details of the submittal.

## Getting Started

To run the chat bot, follow these steps:

1. Clone this repository to your local machine:

```
ssh://git@bitbucket.trimble.tools/cen/submittal_chatbot.git
```


2. Install the required dependencies using `pip` and the `requirements.txt` file:
```
pip install -r requirements.txt
```


3. Contact Mehdi Mahmoodi at mehdi_mahmoodi@trimble.com to obtain the necessary environment file (`.env`) required for running the chat bot.

4. Run the chat bot using Streamlit:
- For the default chat bot version:
  ```
  streamlit run app.py
  ```
- For a version using a document-based model(Recommended):
  ```
  streamlit run app_docmodel.py
  ```

## Usage

1. Upon running the Streamlit app, you will be presented with an interface to upload a construction submittal.
2. Click the "Browse" or "Choose File" button to select the submittal document you want to discuss.
3. After selecting the document, press the "Process" button and wait for the document to be processed.
4. Once the processing is complete, you will see a success message indicating that the document has been successfully loaded.
5. The chat bot will initiate a conversation with you based on the details of the uploaded submittal.
6. You can interact with the bot, ask questions, and discuss the submittal in a conversational manner.


## Contact

If you have any questions, suggestions, or issues, feel free to contact Mehdi Mahmoodi at mehdi_mahmoodi@trimble.com. We welcome your feedback and contributions!


