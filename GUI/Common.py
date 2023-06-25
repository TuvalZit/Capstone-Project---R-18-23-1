# Imports
import tkinter as tk
from tkinter import messagebox
import os


#######################################################################
def setWindowStyle(root, title, size="900x500"):
    # Creating the main window of the GUI
    #########Properties of the main window#########
    root.geometry(size)
    root.title(title)
    root.configure(background="#282828")
    icon_image = tk.PhotoImage(file="resources/shakespeare.png")
    root.iconphoto(True, icon_image)


#################################################################
# Function to show a custom messagebox
def show_custom_messagebox(title, text, parent):
    messagebox.showerror(message=text,
                         icon=messagebox.ERROR, title=title,
                         parent=parent)


##############################################################
# Function to extract text list form parent folder/directory
def extractTextList(parent_folder, textsList):
    for root, _, files in os.walk(parent_folder):
        for filename in files:
            if filename.endswith('.txt'):
                textsList.append(filename)
