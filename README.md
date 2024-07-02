# Chatbot
An LLM chatbot to read documents and answer queries.

## Getting Started

To run the chatbot, follow these steps:

1. Clone this repository to your local machine:

2. Install the required dependencies using `pip` and the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

3. Contact Mehdi Mahmoodi at mehdi-mahmoodi@datuxgroup.com to obtain the necessary environment file (`.env`) required for running the chatbot.

4. Run the chatbot using Streamlit:
    - For the default chatbot version:
        ```bash
        streamlit run app.py
        ```
    - For the document-based model version (Recommended):
        ```bash
        streamlit run app_docmodel.py
        ```

## Usage

1. Upon running the Streamlit app, you will be presented with an interface to upload a construction submittal.
2. Click the "Browse" or "Choose File" button to select the submittal document you want to discuss.
3. After selecting the document, press the "Process" button and wait for the document to be processed.
4. Once the processing is complete, you will see a success message indicating that the document has been successfully loaded.
5. The chatbot will initiate a conversation with you based on the details of the uploaded submittal.
6. You can interact with the bot, ask questions, and discuss the submittal in a conversational manner.

## Contact

If you have any questions, suggestions, or issues, feel free to contact Mehdi Mahmoodi at mehdi-mahmoodi@datuxgroup.com. We welcome your feedback and contributions!
