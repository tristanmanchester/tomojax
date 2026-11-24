"""Run worker commands in process with a pinned input and private array staging."""

from tomojax import progress
from tomojax.cli import execution, products
from tomojax.cli._adapters import execute  # check-public-imports: allow-private
from tomojax.wire import arrays


def admitted(command, value, stage, ref):
    context = execution.ExecutionContext(
        {ref: execution.Lease(value, products.Provenance("open", {}, ()))},
        progress.Control(),
        lambda _: None,
        lambda _: None,
        stage,
    )
    return execute(command, context).value


class Stage:
    def __init__(self, root):
        self.private = root / "staged"
        self.public = root / "published.npy"

    def array(self):
        return execution.ArrayStage(self.private, arrays.Npy(str(self.public), "issued-token"))
