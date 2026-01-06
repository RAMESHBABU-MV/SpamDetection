import streamlit as st
import joblib

st.set_page_config(
    page_title="Spam Classifier",
    page_icon="📧",
    layout="centered"
)

# Load artifacts
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# UI
st.title("📧 Spam Detection Engine")
st.write("Enter a message and let the model decide if it's spam or legit.")

message = st.text_area("Your message here", height=150)

if st.button("Classify"):
    if message.strip() == "":
        st.warning("Message cannot be empty. Feed the model something.")
    else:
        # 🔑 Transform, don’t reshape
        message_vec = vectorizer.transform([message])
        prediction = model.predict(message_vec)[0]

        if prediction == 1:
            st.error("🚨 SPAM detected")
        else:
            st.success("✅ Not Spam (Looks clean)")

        st.write(f"**Prediction Label:** {prediction}")
        st.write("*(0 = Not Spam, 1 = Spam)*")
        st.balloons()

st.markdown("---")
st.caption("Developed by Ramesh S. | Powered by Streamlit")
