import bcrypt

salt = "project-ai-4"


def hash_password(password):
    # Generate a salt and hash the password
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(password.encode("utf-8"), salt)
    return hashed_password


def verify_password(input_password, stored_hashed_password):
    # Verify the input password against the stored hashed password
    return bcrypt.checkpw(input_password.encode("utf-8"), stored_hashed_password)


if __name__ == "__main__":
    # Example usage:
    password_to_hash = "abc"
    hashed_password = hash_password(password_to_hash)

    # Pretend that hashed_password and salt are stored securely (e.g., in a database)

    # Later, when verifying a login attempt:
    input_password = input("Enter password: ")
    if verify_password(input_password, hashed_password):
        print("Password is correct!")
    else:
        print("Password is incorrect.")
