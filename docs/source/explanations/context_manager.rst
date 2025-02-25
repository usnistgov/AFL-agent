The Context Manager Pattern
===========================

The AFL double agent framework employs Python's context manager pattern to create an elegant, intuitive interface for building pipelines. This document explains the conceptual underpinnings of this design choice, how it works internally, and the benefits it provides to developers.

Why Context Managers?
---------------------

Context managers in Python (implemented using the `with` statement) traditionally serve two primary purposes:

1. **Resource Management**: They ensure proper acquisition and release of resources (like file handles or network connections).

2. **State Management**: They temporarily establish a specific state or environment for a block of code.

In the AFL pipeline system, we leverage the second aspect - creating a temporary "context" in which operations are automatically associated with the current pipeline. This design creates a more readable, intuitive API that reduces boilerplate code and makes pipeline construction feel natural in Python.

Here's a simple example of how this pattern improves code readability:

.. code-block:: python

   # Using context manager approach
   with Pipeline(name="MyPipeline") as pipeline:
       MyOperation(input_variable="data", output_variable="processed_data")
       AnotherOperation(input_variable="processed_data", output_variable="final_result")

   # Without context manager - more verbose and error-prone
   pipeline = Pipeline(name="MyPipeline")
   op1 = MyOperation(input_variable="data", output_variable="processed_data")
   pipeline.append(op1)
   op2 = AnotherOperation(input_variable="processed_data", output_variable="final_result")
   pipeline.append(op2)

The Stack-Based Context Design
-----------------------------

At the heart of the context manager implementation is a thread-local stack of pipeline contexts:

.. code-block:: python

   contexts = threading.local()

The implementation in `PipelineContext.py` shows how this stack is managed:

.. code-block:: python

   class PipelineContext:
       """Inherited by Pipeline to allow for context manager abuse"""

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

This design:

1. **Creates Thread Safety**: Using `threading.local()` ensures that different threads maintain separate context stacks, preventing cross-thread interference.

2. **Enables Nesting**: The stack-based approach allows contexts to be nested, with the most recently entered context becoming the "active" one.

3. **Maintains State**: The context stack preserves the relationship between operations and their containing pipeline during construction.

How It Works Conceptually
--------------------------

The pipeline context system works on a simple principle: when you create a pipeline within a `with` statement, that pipeline becomes the "active" context. Any operations created inside that block are automatically added to the active pipeline.

The flow works like this:

1. When a `Pipeline` is created within a `with` statement, it's pushed onto the context stack.
2. While that block executes, the pipeline is accessible as the "current context."
3. When a `PipelineOp` is instantiated, it automatically tries to add itself to the current context.
4. When the `with` block exits, the pipeline is popped from the stack, and any previous context becomes active again.

Here's the code from `PipelineOp.py` that illustrates how operations automatically register themselves:

.. code-block:: python

   def __init__(self,
                name: Optional[str] | List[str] = None,
                input_variable: Optional[str] | List[str] = None,
                output_variable: Optional[str] | List[str] = None,
                input_prefix: Optional[str] | List[str] = None,
                output_prefix: Optional[str] | List[str] = None):
       
       # ... other initialization code ...
       
       try:
           # try to add this object to current pipeline on context stack
           PipelineContext.get_context().append(self)
       except NoContextException:
           # silently continue for those working outside a context manager
           pass

This process creates a natural, hierarchical relationship between pipelines and their operations, making the code structure visually reflect the pipeline structure.

Benefits of the Context Manager Approach
---------------------------------------

This design provides several important benefits:

**Reduced Verbosity**
    Without the context manager, each operation would need to be explicitly added to its pipeline, cluttering the code with repetitive calls.

**Visual Structure**
    The indentation of the `with` block visually indicates which operations belong to which pipeline, enhancing readability.

**Consistent State**
    The context manager ensures that operations are always added to the correct pipeline, reducing the risk of operations being unintentionally omitted or added to the wrong pipeline.

**Graceful Degradation**
    If an operation is created outside any pipeline context, it gracefully handles the situation rather than raising an error, allowing for more flexible usage patterns.

Implementation Details and Considerations
----------------------------------------

There are a few important implementation details to be aware of:

**The NoContextException**
    When attempting to get the current context outside any `with` block, a `NoContextException` is raised. This is handled gracefully in the `PipelineOp` constructor.

.. code-block:: python

   class NoContextException(Exception):
       pass

**Thread Locality**
    Since the context stack is thread-local, pipelines and operations must be created in the same thread. This is typically not an issue but could be important in multithreaded applications.

**Context Management vs. Manual Construction**
    While the context manager provides a convenient way to build pipelines, you can still manually construct pipelines by explicitly adding operations. This flexibility accommodates different programming styles and requirements.

.. code-block:: python

   # Using context manager
   with Pipeline(name="pipeline1") as p1:
       MyOperation(input_variable="x", output_variable="y")

   # Using manual construction
   p2 = Pipeline(name="pipeline2")
   op = MyOperation(input_variable="x", output_variable="y")
   p2.append(op)

Advanced Patterns
-----------------

The context manager design enables several advanced patterns:

**Pipeline Factories**
    Functions that create and return pipelines can leverage the context manager pattern to provide a clean API for building configurable pipeline templates.

.. code-block:: python

   def create_processing_pipeline(data_type, threshold=0.5):
       """Factory function to create standardized processing pipelines"""
       with Pipeline(name=f"{data_type}_processing") as pipeline:
           # Common operations for all data types
           Normalize(input_variable="raw_data", output_variable="normalized_data")
           
           # Conditional operations based on data_type
           if data_type == "image":
               ImageFilter(input_variable="normalized_data", output_variable="filtered_data", 
                           filter_type="gaussian")
               threshold_var = "filtered_data"
           elif data_type == "signal":
               SignalFilter(input_variable="normalized_data", output_variable="filtered_data", 
                            filter_type="lowpass")
               threshold_var = "filtered_data"
           else:
               threshold_var = "normalized_data"
               
           # Final thresholding operation with configurable threshold
           Threshold(input_variable=threshold_var, output_variable="thresholded_data", 
                     threshold=threshold)
           
       return pipeline

   # Usage
   image_pipeline = create_processing_pipeline("image", threshold=0.75)
   signal_pipeline = create_processing_pipeline("signal", threshold=0.25)

**Nested Pipelines**
    While not directly supported in the current implementation, the stack-based design could be extended to support nested pipelines, where sub-pipelines operate within parent pipelines.

.. code-block:: python

   # Conceptual example of how nested pipelines might work
   with Pipeline(name="master_pipeline") as master:
       # Some operations in the master pipeline
       DataLoader(input_variable="file_path", output_variable="raw_data")
       
       # Create a nested pipeline for preprocessing
       with NestedPipeline(name="preprocessing", input_variable="raw_data", 
                            output_variable="preprocessed_data") as preprocess:
           Normalize(input_variable="raw_data", output_variable="normalized")
           RemoveOutliers(input_variable="normalized", output_variable="cleaned")
           # The last output becomes the nested pipeline's output
       
       # Continue with operations in the master pipeline
       ModelPredictor(input_variable="preprocessed_data", output_variable="predictions")

**Dynamic Pipeline Construction**
    The context approach makes it easier to conditionally add operations to a pipeline based on runtime parameters, enhancing flexibility.

.. code-block:: python

   def build_adaptive_pipeline(data_properties):
       """Builds a pipeline that adapts to properties of the data"""
       with Pipeline(name="adaptive_pipeline") as pipeline:
           # Basic operations for all cases
           LoadData(input_variable="data_path", output_variable="raw_data")
           
           # Add preprocessing operations based on data properties
           if data_properties.get("has_missing_values", False):
               ImputeMissingValues(input_variable="raw_data", output_variable="imputed_data")
               current_data = "imputed_data"
           else:
               current_data = "raw_data"
               
           if data_properties.get("needs_normalization", True):
               Normalize(input_variable=current_data, output_variable="normalized_data")
               current_data = "normalized_data"
               
           # Add dimensionality reduction if data has high dimensions
           if data_properties.get("dimensions", 0) > 100:
               PCA(input_variable=current_data, output_variable="reduced_data", 
                   n_components=data_properties.get("target_dimensions", 50))
               current_data = "reduced_data"
           
           # Final output operation
           Analyze(input_variable=current_data, output_variable="results")
           
       return pipeline

   # Usage
   data_props = {
       "has_missing_values": True,
       "needs_normalization": True,
       "dimensions": 500,
       "target_dimensions": 50
   }
   my_pipeline = build_adaptive_pipeline(data_props)

Alternatives Considered
------------------------

The context manager approach was chosen over several alternatives:

**Fluent Builder Pattern**
    A chained method approach (e.g., `pipeline.add_op1().add_op2()`) would be less verbose but wouldn't provide the visual structure of nested blocks.

.. code-block:: python

   # Hypothetical fluent builder approach
   pipeline = (Pipeline(name="MyPipeline")
               .add(DataLoader(input_variable="path", output_variable="data"))
               .add(Process(input_variable="data", output_variable="processed")))

**Explicit Registration**
    Requiring each operation to be explicitly added to a pipeline would be more transparent but more verbose and prone to errors.

.. code-block:: python

   # Explicit registration approach
   pipeline = Pipeline(name="MyPipeline")
   loader = DataLoader(input_variable="path", output_variable="data")
   processor = Process(input_variable="data", output_variable="processed")
   pipeline.add(loader)
   pipeline.add(processor)

**Decorator-Based Approach**
    Using decorators to define pipeline operations would be elegant but might limit flexibility in operation reuse.

.. code-block:: python

   # Hypothetical decorator-based approach
   pipeline = Pipeline(name="MyPipeline")

   @pipeline.operation(input_variable="path", output_variable="data")
   def load_data(dataset):
       # Load data implementation
       return loaded_data

   @pipeline.operation(input_variable="data", output_variable="processed")
   def process_data(dataset):
       # Processing implementation
       return processed_data

Conclusion
----------

The context manager pattern in AFL's pipeline system demonstrates how Python's language features can be leveraged to create intuitive, readable APIs. By using context managers, the framework provides a balance of clarity, flexibility, and conciseness that makes building complex data processing pipelines more manageable.

