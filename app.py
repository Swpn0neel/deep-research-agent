import os
import bcrypt
from pymongo import MongoClient
from dotenv import load_dotenv
import streamlit as st

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("MONGO_DBNAME", "chat_app")

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
users_col = db["users"]

def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt())

def check_password(password, hashed):
    return bcrypt.checkpw(password.encode(), hashed)

def create_user(username, password):
    if users_col.find_one({"username": username}):
        return False
    users_col.insert_one({"username": username, "password": hash_password(password)})
    return True

def authenticate_user(username, password):
    user = users_col.find_one({"username": username})
    return user and check_password(password, user["password"])

st.set_page_config(page_title="Authentication", layout="centered")

st.title("🔑 User Authentication")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.header("Login / Signup")
    tab1, tab2 = st.tabs(["Sign In", "Sign Up"])

    with tab1:
        uname = st.text_input("Username", key="signin_username")
        pwd = st.text_input("Password", type="password", key="signin_password")
        if st.button("Sign In"):
            if authenticate_user(uname, pwd):
                st.session_state.logged_in = True
                st.session_state.username = uname
                st.rerun()
            else:
                st.error("Invalid credentials")

    with tab2:
        uname = st.text_input("New Username", key="signup_username")
        pwd1 = st.text_input("Password", type="password", key="signup_password1")
        pwd2 = st.text_input("Re-enter Password", type="password", key="signup_password2")
        if st.button("Sign Up"):
            if pwd1 != pwd2:
                st.error("Passwords do not match")
            else:
                if create_user(uname, pwd1):
                    st.success("Account created! Please log in.")
                else:
                    st.error("Username already exists")
    st.stop()

st.markdown(f"👤 Logged in as **{st.session_state.username}**")

if st.button("🚪 Logout"):
    st.session_state.logged_in = False
    st.rerun()

st.info("You are now authenticated. This is a blank page.")
