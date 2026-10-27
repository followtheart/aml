"""Client input errors, distinct from provider and internal processing failures."""


class InvalidRequest(ValueError):
    """A deterministic input conflict that requires changing the request."""
