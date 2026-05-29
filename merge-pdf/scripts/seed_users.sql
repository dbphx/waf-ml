-- Replace these bcrypt hashes before production use.
-- Password for both users: ChangeMe123!
INSERT INTO users (email, password_hash, role)
VALUES
    ('admin@example.com', '$2b$12$Imh7hRcGVf0gn0/Bk/5jVuq/UwmT03AVjcXiGariUXQFDgZ8RxYMG', 'admin'),
    ('user@example.com', '$2b$12$Imh7hRcGVf0gn0/Bk/5jVuq/UwmT03AVjcXiGariUXQFDgZ8RxYMG', 'user')
ON CONFLICT (email) DO NOTHING;
