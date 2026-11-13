"""Shared checks for the native accounting audits: bad evidence aborts, it never scores zero."""


class EvidenceError(ValueError):
    """Invalid evidence: abort the instrument, never substitute zero reward."""


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise EvidenceError(f"{name} must be an integer >= {minimum}")
    return value
