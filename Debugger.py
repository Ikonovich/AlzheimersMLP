from pprint import PrettyPrinter


class Debugger:

    printer = PrettyPrinter(indent=4)

    def info(self, message: str):
        self.printer.pprint(message)

    def warn(self, message: str):
        self.printer.pprint(message)

    def error(self, message: str, error: BaseException):
        self.printer.pprint(message)
        self.printer.pprint(error)

    def critical(self, message: str, error: BaseException):
        self.printer.pprint(message)
        self.printer.pprint(error)

