class AeroSenseError(Exception):
    """Base class for all AeroSense application errors."""
    code = "INTERNAL_ERROR"
    status_code = 500

    def __init__(self, message: str, code: str | None = None, status_code: int | None = None):
        super().__init__(message)
        self.message = message
        if code:
            self.code = code
        if status_code:
            self.status_code = status_code


class InvalidAudioError(AeroSenseError):
    code = "INVALID_AUDIO"
    status_code = 422


class UnsupportedFormatError(AeroSenseError):
    code = "UNSUPPORTED_FORMAT"
    status_code = 415


class FileTooLargeError(AeroSenseError):
    code = "FILE_TOO_LARGE"
    status_code = 413


class CorruptAudioError(AeroSenseError):
    code = "CORRUPT_AUDIO"
    status_code = 422


class ModelUnavailableError(AeroSenseError):
    code = "MODEL_UNAVAILABLE"
    status_code = 503


class InferenceError(AeroSenseError):
    code = "INFERENCE_FAILED"
    status_code = 500


class DatabaseError(AeroSenseError):
    code = "DATABASE_ERROR"
    status_code = 500


class NotFoundError(AeroSenseError):
    code = "NOT_FOUND"
    status_code = 404
