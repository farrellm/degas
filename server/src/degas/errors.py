"""The base of Degas's own exceptions.

Each part of the server defines its errors next to the code that raises them (`MediaError`,
`SpecError`, `SessionError` …); they all derive from `DegasError`. `degas.api.errors` maps
the ones a request can meet to HTTP statuses.
"""


class DegasError(Exception):
    """Something Degas can explain: its message is written for the person using it."""
