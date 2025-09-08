# Streamlit User Authentication with MongoDB

This is a simple yet secure user authentication boilerplate built with Streamlit. It provides a clean interface for user registration (Sign Up) and login (Sign In), with user credentials securely stored in a MongoDB database. Passwords are hashed using **bcrypt** to ensure they are not stored in plaintext.

---

## ✨ Features

- **User Registration:** New users can sign up for an account.
- **User Login:** Existing users can log in to access protected content.
- **Secure Password Storage:** Passwords are hashed using **bcrypt** before being stored in the database.
- **Session Management:** Utilizes Streamlit's built-in `session_state` to manage user login status.
- **MongoDB Integration:** Seamlessly connects to a MongoDB database to store and retrieve user data.
- **Environment-based Configuration:** Uses a `.env` file to manage sensitive credentials like database URIs.

---

## 🛠️ Technologies Used

- **Framework:** [Streamlit](https://streamlit.io/)
- **Database:** [MongoDB](https://www.mongodb.com/)
- **Python Libraries:**
  - `pymongo` (for MongoDB interaction)
  - `bcrypt` (for password hashing)
  - `python-dotenv` (for managing environment variables)
  - `streamlit`

---

## 🚀 Getting Started

Follow these steps to get the project up and running on your local machine.

### Prerequisites

- Python 3.8+
- A MongoDB Atlas account or a local MongoDB instance.

### Installation

1.  **Save the script:**
    Save the provided code as a Python file (e.g., `app.py`).

2.  **Create a virtual environment (recommended):**

    ```bash
    # Create the environment
    python -m venv venv

    # Activate the environment
    # On Windows:
    # venv\Scripts\activate
    # On macOS/Linux:
    # source venv/bin/activate
    ```

3.  **Install the required packages:**
    Create a `requirements.txt` file with the following content:

    ```txt
    streamlit
    pymongo
    bcrypt
    python-dotenv
    dnspython
    ```

    Then, install the packages using pip:

    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables:**
    Create a file named `.env` in the same directory as your script and add your MongoDB connection details:
    ```.env
    MONGO_URI="your_mongodb_connection_string"
    MONGO_DBNAME="your_database_name" # Optional, defaults to "chat_app"
    ```
    Replace `"your_mongodb_connection_string"` with your actual MongoDB URI (e.g., from MongoDB Atlas).

---

## ▶️ How to Run

Once you have completed the setup, run the Streamlit application using the following command in your terminal:

```bash
streamlit run app.py
```
