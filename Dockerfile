FROM python:3.11-slim

WORKDIR /app

# Copy project and install (editable) with its runtime dependencies.
COPY . .
RUN pip install --no-cache-dir -e .

# Default: run the end-to-end synthetic-data demo.
CMD ["python", "examples/demo.py"]
