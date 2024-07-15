# Use an official Python runtime as a parent image
FROM public.ecr.aws/lambda/python:3.8

# Set the working directory in the container
WORKDIR /var/task

# Copy the requirements file and install dependencies
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Copy the deepgaze_pytorch directory if it is local
# Uncomment and modify the next line if deepgaze_pytorch is a local directory
COPY deepgaze_pytorch ./deepgaze_pytorch

# Copy the centerbias_mit1003.npy file
COPY centerbias_mit1003.npy ./centerbias_mit1003.npy

# Copy the rest of the application code
COPY deepgaze_api.py ./deepgaze_api.py

# Set MPLCONFIGDIR to a writable directory
ENV MPLCONFIGDIR=/tmp

# Set the CMD to your Lambda function handler
CMD ["deepgaze_api.handler"]
