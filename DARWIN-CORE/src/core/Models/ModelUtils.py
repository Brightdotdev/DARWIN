class DownloadError(Exception):
    def __init__(self, message: str, hint: str = "", recoverable: bool = True):
        self.hint = hint
        self.recoverable = recoverable
        super().__init__(message)


class IntegrityError(Exception):
    pass
