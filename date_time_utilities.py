from datetime import datetime, timedelta, tzinfo
import pytz


class DateTimeUtils:
    def __init__(self, dt=None, tz=None):
        self.dt = dt or datetime.now()
        self.tz = tz or pytz.UTC

    def to_timezone(self, tz):
        """
        Converts the datetime to the specified timezone.

        Args:
            tz (str or tzinfo): Timezone identifier or tzinfo object.

        Returns:
            DateTimeUtils: Updated DateTimeUtils object.

        Examples:
            - to_timezone('US/Eastern'): Converts the datetime to US/Eastern timezone.
            - to_timezone(pytz.timezone('Asia/Tokyo')): Converts the datetime to Asia/Tokyo timezone.
        """
        if isinstance(tz, str):
            try:
                tz = pytz.timezone(tz)
            except pytz.UnknownTimeZoneError:
                raise ValueError("Invalid timezone: {}".format(tz))
        elif not isinstance(tz, tzinfo):
            raise TypeError("Invalid timezone type: {}".format(type(tz)))

        self.dt = self.dt.astimezone(tz)
        self.tz = tz
        return self

    def to_datetime(self):
        """Returns the datetime object."""
        if isinstance(self.dt, datetime):
            return self.dt
        elif isinstance(self.dt, str):
            try:
                return datetime.fromisoformat(self.dt)
            except ValueError:
                pass
            try:
                return datetime.strptime(self.dt, '%Y-%m-%d %H:%M:%S')
            except ValueError:
                pass
        raise TypeError("Unsupported type for conversion to datetime.")

    def to_string(self, format='%Y-%m-%d %H:%M:%S'):
        """Returns the datetime as a string in the specified format."""
        return self.dt.strftime(format)

    def to_timestamp(self):
        """Returns the timestamp representation of the datetime."""
        return self.dt.timestamp()

    @classmethod
    def from_string(cls, dt_str, tz=None, format='%Y-%m-%d %H:%M:%S'):
        """Creates a DateTimeUtils instance from a string."""
        dt = datetime.strptime(dt_str, format)
        return cls(dt, tz)

    @classmethod
    def from_timestamp(cls, timestamp, tz=None):
        """Creates a DateTimeUtils instance from a timestamp."""
        dt = datetime.fromtimestamp(timestamp)
        return cls(dt, tz)

    def __str__(self):
        """Returns a string representation of the DateTimeUtils object."""
        return self.to_string()

    def __repr__(self):
        """Returns a string representation of the DateTimeUtils object."""
        return self.to_string()

    def __add__(self, other):
        """Adds a timedelta or a number of seconds, minutes, hours, days to the datetime."""
        if isinstance(other, timedelta):
            self.dt += other
        elif isinstance(other, (int, float)):
            self.dt += timedelta(seconds=other)
        else:
            raise TypeError("Unsupported operand type for +: {}".format(type(other)))
        return self

    def __sub__(self, other):
        """Subtracts a timedelta or a number of seconds, minutes, hours, days from the datetime."""
        if isinstance(other, timedelta):
            self.dt -= other
        elif isinstance(other, (int, float)):
            self.dt -= timedelta(seconds=other)
        else:
            raise TypeError("Unsupported operand type for -: {}".format(type(other)))
        return self

    def __eq__(self, other):
        """Checks if two DateTimeUtils objects are equal."""
        if isinstance(other, DateTimeUtils):
            return self.dt == other.dt and self.tz == other.tz
        return NotImplemented

    def __lt__(self, other):
        """Checks if a DateTimeUtils object is less than another."""
        if isinstance(other, DateTimeUtils):
            return self.dt < other.dt
        return NotImplemented

    def __gt__(self, other):
        """Checks if a DateTimeUtils object is greater than another."""
        if isinstance(other, DateTimeUtils):
            return self.dt > other.dt
        return NotImplemented

    def __le__(self, other):
        """Checks if a DateTimeUtils object is less than or equal to another."""
        if isinstance(other, DateTimeUtils):
            return self.dt <= other.dt
        return NotImplemented

    def __ge__(self, other):
        """Checks if a DateTimeUtils object is greater than or equal to another."""
        if isinstance(other, DateTimeUtils):
            return self.dt >= other.dt
        return NotImplemented

    def difference(self, other, unit='seconds'):
        """Calculates the difference between two DateTimeUtils objects in the specified unit."""
        if isinstance(other, DateTimeUtils):
            delta = self.dt - other.dt
            if unit == 'seconds':
                return delta.total_seconds()
            elif unit == 'minutes':
                return delta.total_seconds() / 60
            elif unit == 'hours':
                return delta.total_seconds() / 3600
            elif unit == 'days':
                return delta.days
            else:
                raise ValueError("Unsupported unit for difference calculation.")
        raise TypeError("Unsupported operand type for difference calculation.")


if __name__ == "__main__":
    # Create a DateTimeUtils object with the current datetime in UTC
    dt = DateTimeUtils()

    # Convert the datetime to a different timezone
    dt.to_timezone('US/Eastern')

    # Print the datetime as a string
    print(dt.to_string())  # Output: 2023-06-24 09:30:00

    # Add 1 hour to the datetime
    dt += timedelta(hours=1)

    # Print the updated datetime
    print(dt.to_string())  # Output: 2023-06-24 10:30:00

    # Convert the datetime back to UTC
    dt.to_timezone('UTC')

    # Print the datetime as a timestamp
    print(dt.to_timestamp())  # Output: 1674532200.0

    # Create a DateTimeUtils object from a string in a different format
    dt2 = DateTimeUtils.from_string('2023-06-24T15:30:00', tz='US/Pacific')

    # Print the datetime as a string
    print(dt2)  # Output: 2023-06-24 15:30:00

    # Perform arithmetic operations
    dt3 = DateTimeUtils()  # Current datetime
    dt3 += 2.5  # Add 2.5 seconds
    dt3 -= timedelta(minutes=10)  # Subtract 10 minutes
    print(dt3.to_string())  # Output: Updated datetime with seconds and minutes adjusted

