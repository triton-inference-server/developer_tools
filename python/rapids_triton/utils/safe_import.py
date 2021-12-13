class UnavailableError(Exception):
    '''Error thrown if a symbol is unavailable due to an issue importing it'''

class ImportReplacement:
    """A class to be used in place of an importable symbol if that symbol
    cannot be imported

    Parameters
    ----------
    symbol: str
        The name or import path to be used in error messages when attempting to
        make use of this symbol. E.g. "some_pkg.func" would result in an
        exception with message "some_pkg.func could not be imported"
    """
    def __init__(self, symbol):
        self._msg = f'{symbol} could not be imported'

    def __getattr__(self, name):
        raise UnavailableError(self._msg)

    def __call__(self, *args, **kwargs):
        raise UnavailableError(self._msg)
