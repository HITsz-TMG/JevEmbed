class JevEmbedError(Exception):
    """Base public error."""


class ValidationError(JevEmbedError, ValueError):
    pass


class BackendError(JevEmbedError):
    pass
