
import sys
import traceback


class HealthcareFraudException(Exception):
    """Custom exception with the original error location."""

    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)

        _, _, exc_tb = error_detail.exc_info()

        if exc_tb is not None:
            filename = exc_tb.tb_frame.f_code.co_filename
            line_number = exc_tb.tb_lineno

            self.error_message = (
                f"Error in [{filename}] "
                f"at line [{line_number}]: "
                f"{error_message}"
            )
        else:
            self.error_message = str(error_message)

    def __str__(self):
        return self.error_message
