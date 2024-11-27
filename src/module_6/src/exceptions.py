class UserNotFoundException(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class PredictionException(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)
