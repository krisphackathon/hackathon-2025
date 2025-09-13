curl -X GET \
     -H "Authorization: Bearer $(gcloud auth print-access-token)" \
     "https://us-documentai.googleapis.com/v1/projects/krisp-hackathon/locations/us/processors"