from .json_reporter import JsonReporter
from .csv_reporter import CsvReporter
from .latex_reporter import LatexReporter
from .wandb_reporter import WandbReporter

REPORTER_REGISTRY = {
    "json": JsonReporter,
    "csv": CsvReporter,
    "latex": LatexReporter,
    "wandb": WandbReporter,
}
