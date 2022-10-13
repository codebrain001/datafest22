# Start by pulling the python image
FROM python:3.9.10
# Set and create the working directory to /app
WORKDIR /app
# Copy the current directory contents into /app
COPY . /app
# Informs docker that this container should listen to network port 8501 atruntime
EXPOSE 8501
# Install the dependencies and packages in the requirements file
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
# Executing the container
CMD ["streamlit", "run", "app.py"]