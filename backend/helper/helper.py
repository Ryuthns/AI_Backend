import bcrypt
import threading
import queue

salt = "project-ai-4"


def hash_password(password):
    # Generate a salt and hash the password
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(password.encode("utf-8"), salt)
    return hashed_password


def verify_password(input_password, stored_hashed_password):
    # Verify the input password against the stored hashed password
    return bcrypt.checkpw(input_password.encode("utf-8"), stored_hashed_password)


def select(*queues):
    combined = queue.Queue(maxsize=0)

    def listen_and_forward(queue):
        while True:
            combined.put((queue, queue.get()))

    for q in queues:
        t = threading.Thread(target=listen_and_forward, args=(q,))
        t.daemon = True
        t.start()
    while True:
        yield combined.get()


def main():
    c1 = queue.Queue(maxsize=0)
    c2 = queue.Queue(maxsize=0)
    quit = queue.Queue(maxsize=0)

    def func1():
        for i in range(10):
            c1.put(i)
        quit.put(0)

    threading.Thread(target=func1).start()

    def func2():
        for i in range(2):
            c2.put(i)

    threading.Thread(target=func2).start()

    for which, msg in select(c1, c2, quit):
        if which is c1:
            print("Received value from c1")
        elif which is c2:
            print("Received value from c2")
        elif which is quit:
            print("Received value from quit")
            return


if __name__ == "__main__":
    # Example usage:
    main()
