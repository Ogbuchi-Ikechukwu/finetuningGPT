#loading dependencies for class based architecture
import os
import json
import signal
import datetime
import time
from openai import OpenAI

class FineTuning:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
    #class function to upload files and print id
    def upload_datasets(self, training_file_id, validation_file_id):
        self.training_file_id = training_file_id
        self.validation_file_id = validation_file_id
        print(f"Training File ID: {self.training_file_id}")
        print(f"Validation File ID: {self.validation_file_id}")
    #class function to initiate fine tuning with parameters defined
    def initiate_fine_tuning(self):
        response = self.client.fine_tuning.jobs.create(
            training_file=self.training_file_id, 
            validation_file=self.validation_file_id,
            model="davinci-002", 
            hyperparameters={
                "n_epochs": 15,
                "batch_size": 3,
                "learning_rate_multiplier": 0.3
            }
        )
        self.job_id = response.id
        self.status = response.status
        print(f'Fine-tuning model with jobID: {self.job_id}.')
    #class functions to see progress in python and check if it still in connection
    def stream_events(self):
        print(f"Streaming events for the fine-tuning job: {self.job_id}")
        signal.signal(signal.SIGINT, self.signal_handler)
        events = self.client.fine_tuning.jobs.list_events(fine_tuning_job_id=self.job_id)
        try:
            for event in events:
                print(f'{datetime.datetime.fromtimestamp(event.created_at)} {event.message}')
        except Exception:
            print("Stream interrupted (client disconnected).")

    def signal_handler(self, sig, frame):
        status = self.client.fine_tuning.jobs.retrieve(self.job_id).status
        print(f"Stream interrupted. Job is still {status}.")
    #class function to check progress of job
    def wait_for_completion(self):
        status = self.client.fine_tuning.jobs.retrieve(self.job_id).status
        if status not in ["succeeded", "failed"]:
            print(f"Job not in terminal status: {status}. Waiting.")
            while status not in ["succeeded", "failed"]:
                time.sleep(2)
                status = self.client.fine_tuning.jobs.retrieve(self.job_id).status
                print(f"Status: {status}")
        else:
            print(f"Fine-tune job {self.job_id} finished with status: {status}")
    #class functions for validating and testing models
    def validate_model(self):
        result = self.client.fine_tuning.jobs.list()
        fine_tuned_model = result.data[0].fine_tuned_model
        print(f"Fine-tuned model: {fine_tuned_model}")

    def test_model(self):
        new_prompts = [
            "Given age of patient as 80 and gender as F, predict API severity?",
            "Given only the values of payment source as private, birth weight as 5kg, predict apn severity?"
        ]
        for prompt in new_prompts:
            answer = self.client.completions.create(
                model=fine_tuned_model,
                prompt=prompt
            )
            print(answer.choices[0].text)

# Defining parameters needed to initialise class functions
api_key = os.environ['OPENAI_API_KEY']
client = OpenAI(api_key=api_key)

training_file_id = client.files.create(
    file=open("training_data.jsonl", "rb"),
    purpose="fine-tune"
)

validation_file_id = client.files.create(
    file=open("validation_data.jsonl", "rb"),
    purpose="fine-tune"
)
#Initialising class with class function
fine_tuner = FineTuning(api_key)
fine_tuner.upload_datasets(training_file_id.id, validation_file_id.id)
fine_tuner.initiate_fine_tuning()
fine_tuner.stream_events()
fine_tuner.wait_for_completion()
fine_tuner.validate_model()
fine_tuner.test_model()
