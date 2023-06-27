import re
from datetime import datetime

class DataValidator:
    def __init__(self, data):
        self.data = data

    def validate_email(self):
        """Validate email addresses using regex."""
        try:
            pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
            match = re.match(pattern, self.data)
            if match:
                return True
            else:
                return False
        except (TypeError, AttributeError):
            return False

    def validate_phone_number(self):
        """Validate phone numbers using regex."""
        try:
            pattern = r'^\d{3}[-.\s]?\d{3}[-.\s]?\d{4}$'
            if re.match(pattern, self.data):
                return True
            else:
                return False
        except TypeError:
            return False

    def validate_url(self):
        """Validate URLs using regex."""
        try:
            pattern = r'^(https?|ftp)://[^\s/$.?#].[^\s]*$'
            if re.match(pattern, self.data):
                return True
            else:
                return False
        except TypeError:
            return False
        except re.error:
            return False

    def validate_address(self):
        """Validate addresses using regex."""
        try:
            pattern = r'^\d+\s[A-Za-z]+\s[A-Za-z]+'
            match = re.match(pattern, self.data)
            if match:
                return True
            else:
                return False
        except (TypeError, re.error) as e:
            print(f"Error occurred while validating address: {e}")
            return False

    def validate_date(self):
        """Validate dates using regex."""
        patterns = [
            r'\d{2}-\d{2}-\d{4}',
            r'\d{4}-\d{2}-\d{2}',
            r'\d{2}/\d{2}/\d{4}',
            r'\d{4}/\d{2}/\d{2}'
        ]
        for pattern in patterns:
            try:
                if re.match(pattern, self.data):
                    return True
            except re.error:
                pass  # Handle regex errors gracefully
        return False

    def validate_datetime(self, format="%Y-%m-%d"):
        """Validate datetimes using a specific format. If no format is provided, it uses the default format."""
        try:
            datetime.strptime(self.data, format)
            return True
        except (ValueError, TypeError) as e:
            return False

    def validate_password(self, min_length=8, require_lowercase=True, require_uppercase=True,
                          require_numeric=True, require_special_chars=False):
        """Validate passwords."""
        try:
            # Rule 1: Minimum length
            if len(self.data) < min_length:
                raise ValueError("Password length should be at least {} characters.".format(min_length))

            # Rule 2: Lowercase requirement
            if require_lowercase and not re.search(r"[a-z]", self.data):
                raise ValueError("Password should contain at least one lowercase letter.")

            # Rule 3: Uppercase requirement
            if require_uppercase and not re.search(r"[A-Z]", self.data):
                raise ValueError("Password should contain at least one uppercase letter.")

            # Rule 4: Numeric requirement
            if require_numeric and not re.search(r"\d", self.data):
                raise ValueError("Password should contain at least one numeric digit.")

            # Rule 5: Special characters requirement
            if require_special_chars and not re.search(r'[!@#$%^&*(),.?":{}|<>]', self.data):
                raise ValueError("Password should contain at least one special character.")

        except ValueError as e:
            return str(e)  # Return the specific validation error message

        return True  # Password is valid

    def validate_username(self, min_length=6, max_length=20):
        """Validate usernames."""
        try:
            if len(self.data) < min_length or len(self.data) > max_length:
                return False

            # Username validation logic here
            pattern = r'^[a-zA-Z0-9_]+$'
            if re.match(pattern, self.data):
                return True
            else:
                return False

        except TypeError:
            return False

    def search_words(self, search_query, case_sensitive=False):
        """Search for words in the data using regex."""
        try:
            pattern = fr'\b{re.escape(search_query)}\b'
            flags = 0 if case_sensitive else re.IGNORECASE
            matches = re.findall(pattern, self.data, flags=flags)
            return matches
        except re.error as e:
            print(f"Regex error occurred: {e}")
            return []


if __name__ == "__main__":
    # Example 1: Validating an email address
    email = "example@gmail.com"
    validator = DataValidator(email)
    if validator.validate_email():
        print(f"{email} is a valid email address.")
    else:
        print(f"{email} is not a valid email address.")

    # Example 2: Validating a phone number
    phone_number = "123-456-7890"
    validator = DataValidator(phone_number)
    if validator.validate_phone_number():
        print(f"{phone_number} is a valid phone number.")
    else:
        print(f"{phone_number} is not a valid phone number.")

    # Example 3: Validating a URL
    url = "https://www.example.com"
    validator = DataValidator(url)
    if validator.validate_url():
        print(f"{url} is a valid URL.")
    else:
        print(f"{url} is not a valid URL.")

    # Example 4: Validating an address
    address = "123 Main Street"
    validator = DataValidator(address)
    if validator.validate_address():
        print(f"{address} is a valid address.")
    else:
        print(f"{address} is not a valid address.")

    # Example 5: Validating a date
    date = "06/26/2023"
    validator = DataValidator(date)
    if validator.validate_date():
        print(f"{date} is a valid date.")
    else:
        print(f"{date} is not a valid date.")

    # Example 6: Validating a datetime
    datetime_str = "2023-06-26"
    validator = DataValidator(datetime_str)
    if validator.validate_datetime("%Y-%m-%d"):
        print(f"{datetime_str} is a valid datetime.")
    else:
        print(f"{datetime_str} is not a valid datetime.")

    # Example 7: Validating a password
    password = "Abc12345!"
    validator = DataValidator(password)
    try:
        validator.validate_password(min_length=8, require_lowercase=True, require_uppercase=True,
                                    require_numeric=True, require_special_chars=True)
        print(f"{password} is a valid password.")
    except ValueError as e:
        print(f"Invalid password: {str(e)}")

    # Example 8: Validating a username
    username = "john_doe123"
    validator = DataValidator(username)
    if validator.validate_username(min_length=6, max_length=20):
        print(f"{username} is a valid username.")
    else:
        print(f"{username} is not a valid username.")

    # Example 9: Searching for words
    text = "Hello, hello, hello world!"
    validator = DataValidator(text)
    matches = validator.search_words("hello")
    print(f"Found matches: {matches}")
