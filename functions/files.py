import os
import shutil
import gradio as gr
from .document_processor import process_pdf, delete_from_chroma

UPLOAD_FOLDER = "data"


def upload_file(file):
    shutil.copy(file, UPLOAD_FOLDER)

    local_file = os.path.join(UPLOAD_FOLDER, file.split("/")[-1])
    process_pdf(local_file)

    gr.Info("File Uploaded!")


def delete_files(files):
    for file in files:
        os.remove(file)

        local_file = os.path.join(UPLOAD_FOLDER, file.split("/")[-1])
        delete_from_chroma(local_file)

    gr.Info("Files Deleted!")