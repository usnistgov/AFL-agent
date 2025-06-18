# AFL.double_agent Test Suite

This directory contains tests for the AFL.double_agent module. The test suite is organized to encourage high test coverage and proper testing practices.

## Structure

- **`test_\*`**: Unit and integration tests for individual components, and interacting components
- **`utils.py`**: Mock classes and helper methods for tests

## Running Tests

### Basic Test Execution

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=AFL.double_agent

# Run a specific test file
pytest tests/unit/test_pipeline.py

# Run a specific test function
pytest tests/unit/test_pipeline.py::test_pipeline_creation
```

### Using Tox for Automated Testing

```bash
# Run tests in all supported Python versions
tox

# Run only linting checks
tox -e lint

# Run only formatting
tox -e format

# Run only documentation build
tox -e docs
```

## Test Coverage

The test suite aims for 100% code coverage. Coverage reports are generated automatically when running tests with the `--cov` option.

```bash
# Generate HTML coverage report
pytest --cov=AFL.double_agent --cov-report=html

# Open the coverage report
open htmlcov/index.html
```

## Writing Tests

### Unit Tests

Unit tests should:
- Test a single component in isolation
- Use mocks when necessary to isolate from external dependencies
- Be fast and deterministic
- Include both normal and edge cases

Example:
```python
@pytest.mark.unit
def test_pipeline_append():
    """Test adding an operation to a Pipeline."""
    pipeline = Pipeline("TestPipeline")
    op = MockPipelineOp(input_variable="x", output_variable="y")
    
    pipeline.append(op)
    
    assert len(pipeline.ops) == 1
    assert pipeline.ops[0] == op
```

### Integration Tests

Integration tests should:
- Test interaction between multiple components
- Use real (or realistic) data
- Verify end-to-end functionality

Example:
```python
@pytest.mark.integration
def test_pipeline_execution(synthetic_dataset):
    """Test executing a pipeline with multiple operations."""
    pipeline = Pipeline("TestPipeline")
    pipeline.append(MockPreprocessor(input_variable="measurement", output_variable="processed"))
    pipeline.append(MockExtractor(input_variable="processed", output_variable="features"))
    
    result = pipeline.calculate(synthetic_dataset)
    
    assert "processed" in result
    assert "features" in result
    # Additional assertions...
```

## Property-Based Testing with Hypothesis

For comprehensive testing of complex operations, use Hypothesis for property-based testing.

Example:
```python
from hypothesis import given, strategies as st

@given(data=st.data())
def test_pipeline_with_hypothesis(data):
    """Test Pipeline with randomized data using Hypothesis."""
    samples = data.draw(st.integers(min_value=1, max_value=10))
    points = data.draw(st.integers(min_value=10, max_value=100))
    
    # Create test data based on the drawn parameters
    # ...
    
    # Test assertions
    # ...
``` 