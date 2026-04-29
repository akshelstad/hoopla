class KeywordSearchError(Exception):
    pass


class SemanticSearchError(Exception):
    pass


class DataLoadError(KeywordSearchError):
    pass


class MissingDataFileError(DataLoadError):
    pass


class InvalidDataFormatError(DataLoadError):
    pass


class MissingIndexError(KeywordSearchError):
    pass


class CorruptIndexError(KeywordSearchError):
    pass


class InvalidQueryError(KeywordSearchError):
    pass


class InvalidDocumentError(KeywordSearchError):
    pass
