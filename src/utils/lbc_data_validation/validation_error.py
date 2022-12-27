class DataValidationError(Exception):
    """raised when data are not valid"""

    def __init__(self, error_type, severity, feature, message):
        self.error_type = error_type
        self.severity = severity
        self.feature = feature
        super().__init__(message)

    def __str__(self):
        return (
            f"\n\n ** A VALIDITY ISSUE OCCURED WHEN CHECKING DATA ** \n"
            f"type(s) : {self.error_type}\n"
            f"Severity : {self.severity} \n"
            f"Feature : {self.feature} \n"
            f"Description : {super().__str__()}"
        )
