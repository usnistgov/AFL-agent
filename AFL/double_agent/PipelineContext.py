import threading
from typing import Any, List


class NoContextException(Exception):
    pass


class PipelineContext:
    """Base class that provides context manager functionality for pipeline operations.
    
    This class implements a thread-local context stack pattern that allows pipeline
    operations to be associated with their parent pipeline. When a Pipeline instance
    is used as a context manager (with statement), it pushes itself onto the context
    stack, making it available to operations created within that context.
    
    This pattern enables a more intuitive API where operations can automatically
    associate with the currently active pipeline without explicit references.
    
    Examples
    --------
    >>> with Pipeline(name="my_pipeline") as pipe:
    ...     # Operations created here can access the pipeline via get_context()
    ...     op = SomeOperation(input_var="data", output_var="result")
    ...     # op can now reference the pipeline context
    
    Notes
    -----
    This implementation uses thread-local storage to ensure thread safety when
    multiple pipelines are being constructed simultaneously in different threads.
    
    See Also
    --------
    Pipeline : Main container class that inherits this context functionality
    """

    contexts = threading.local()

    def __enter__(self):
        type(self).get_contexts().append(self)
        return self

    def __exit__(self, typ, value, traceback):
        type(self).get_contexts().pop()

    @classmethod
    def get_contexts(cls) -> List:
        if not hasattr(cls.contexts, "stack"):
            cls.contexts.stack = []
        return cls.contexts.stack

    @classmethod
    def get_context(cls) -> Any:
        """Return the deepest context on the stack."""
        try:
            return cls.get_contexts()[-1]
        except IndexError:
            raise NoContextException("No context on context stack")
