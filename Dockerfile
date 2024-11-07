# Use Python 3.9 as the base image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy Python script and CSV file to the container
COPY mca-ranklist-analysis-2024.py /app/
COPY cleaned_rank_list.csv /kaggle/input/mca-ranklist-analysis/
COPY cleaned_prov_allot.csv /kaggle/input/mca-ranklist-analysis/

# Install necessary Python packages
RUN pip install pandas scikit-learn

# Run the Python script
CMD ["python", "/app/mca-ranklist-analysis-2024.py"]
