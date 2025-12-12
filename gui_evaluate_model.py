#!/usr/bin/python3
import pathlib
import tkinter as tk
import tkinter.ttk as ttk
import pygubu
import os
import subprocess
import re
from gui_evaluate_modelui import gui_evaluate_modelUI

PROJECT_PATH = pathlib.Path(__file__).parent
PROJECT_UI = PROJECT_PATH / "pygubu_designer/evaluate.ui"
RESOURCE_PATHS = [PROJECT_PATH]


class gui_evaluate_model(gui_evaluate_modelUI):
    def __init__(self, master=None):
        super().__init__(
            master,
            project_ui=PROJECT_UI,
            resource_paths=RESOURCE_PATHS,
            translator=None,
            on_first_object_cb=None
        )
        self.builder.connect_callbacks(self)
        self.modelInput : pygubu.widgets.PathChooserInput = self.builder.get_object("modelInput")
        self.resultLabel : ttk.Label = self.builder.get_object("resultLabel")
        self.reviewInput : tk.Text = self.builder.get_object("reviewInput")

    def __change_result_text(self, text : str):
        self.resultLabel.config(text=text)

    def __run_evaluation(self):
        model_path = self.modelInput.entry.get()
        reviewTextInput = self.reviewInput.get("1.0", tk.END)

        result = subprocess.run(
            ["python", "evaluate_model.py", "-m", "single", "-i", reviewTextInput, "-im", model_path],
            capture_output=True,
            text=True,
            check=True
        )

        val = result.stdout.strip()
        val_clean = re.sub(r'\x1b\[[0-9;]*[mHJ]', '', val)
        return float(val_clean)
        

    def e_evaluate_button_press(self):
        evaluate_class = lambda val : "Positive" if val > 0.5 else "negative"
        val = self.__run_evaluation()
        self.__change_result_text(f"{str(val)} = {evaluate_class(val)}")
        pass


if __name__ == "__main__":
    app = gui_evaluate_model()
    app.run()
