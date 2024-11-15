import streamlit as st
import pandas as pd
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')


if 'emails' not in st.session_state:
    st.session_state.emails = [
        {"Subject": "Invoice Due", "Sender": "bob@invoice.com", "Content": """Hi User,

Your invoice is due soon.

Please complete the payment by tomorrow
                                                                           
Regards,
Invoice Company""", "Label": "Ham"},
        {"Subject": "Meeting Reminder", "Sender": "carol@company.com", "Content": """Hi team,

This is a reminder for tomorrow's meeting at 10 AM...

Regards""", "Label": "Ham"},
        {"Subject": "Congratulations for a free iPhone", "Sender": "promotions@winbig-prizes.com", "Content": """Dear Valued User,

We are thrilled to inform you that you have been selected as our lucky winner! 🎉 As part of our annual giveaway, you now have the chance to receive a brand new iPhone 14 — completely FREE!

All you need to do is click the link below and complete a short survey to claim your prize:

👉 **[Claim My iPhone 14](http://winbig-prizes.com/claim-now)**

Hurry, you only have **24 hours** to claim your reward before it expires! Don’t miss out on this once-in-a-lifetime opportunity.

Congratulations once again, and we look forward to seeing you with your new iPhone!

Best regards,  
The WinBig Prizes Team""", "Label": "Spam"}
    ]

model_pkl_file = "model.pkl"
vectorizer_pkl_file = "vectorizer.pkl"

with open(model_pkl_file, 'rb') as file:
    model = pickle.load(file)
with open(vectorizer_pkl_file, 'rb') as file:
    vectorizer = pickle.load(file)

ps = PorterStemmer()


def check_spam(body):
    stop_words = set(stopwords.words('english'))

    preprocessed_email = [ps.stem(word) for word in word_tokenize(body.lower()) if word.isalpha()
                          and word not in stop_words]  # perform stemming on every word

    corpus = ' '.join(preprocessed_email)  # get the stemmed sms
    features = vectorizer.transform([corpus]).toarray()

    return model.predict(features)


def main():
    st.title("Email Classification Demo")

    with st.sidebar:
        st.header("Craft your email")
        st.caption("Enter the fields below and hit send to check the spam filter in action")

        with st.form(key="email"):
            from_email = st.text_input("From")
            subject = st.text_input("Subject")
            body = st.text_area("Email Body")
            submit = st.form_submit_button("Send")

    if submit:
        is_spam = check_spam(body)[0]

        # Add the new email to the session state emails list
        new_email = {"Subject": subject, "Sender": from_email, "Content": body, "Label": "Spam" if is_spam else "Ham"}
        st.session_state.emails.append(new_email)

        if is_spam:
            st.warning("Spam email detected!")
        else:
            st.success("Email added successfully!")

    email_df = pd.DataFrame(st.session_state.emails)

    # Separate the spam and ham emails
    spam_emails = email_df[email_df['Label'] == 'Spam']
    ham_emails = email_df[email_df['Label'] == 'Ham']

    tab1, tab2 = st.tabs(['Inbox', 'Spam'])

    with tab1:
        st.subheader("Inbox")
        if ham_emails.empty:
            st.write("No emails")
        else:
            st.write(ham_emails[["Subject", "Sender", "Content"]])

    with tab2:
        st.subheader("Spam")
        if spam_emails.empty:
            st.write("No spam emails.")
        else:
            st.write(spam_emails[["Subject", "Sender", "Content"]])


if __name__ == '__main__':
    main()
