#Imports
import tkinter as tk
from tkinter import ttk
from Backend.GetEmbedding import twoEmbeddings, creatEmbedding
from ChooseDocuments import ChooseDocScreen
from Common import setWindowStyle
##################################################################################
#Function to open the Choose Document Screen
def show_choose_doc_screen():
    root.destroy()
    ChooseDocScreen()
##################################################################################
# Creating the main window of the GUI
root = tk.Tk()
#########Properties of the main window#########
setWindowStyle(root, "PART : Pre-trained Authorship Embedding Transformer")
###################################################################
#########Content of the main window#########
# header
header1 = tk.Label(root, text="Wellcome to:", font=('Ariel', 38), bg="#282828", fg="whitesmoke")
header1.pack(padx=20, pady=5)
txt = "PART\nPre-trained Authorship\nRepresentation Transformer"
header2 = tk.Label(root, text=txt, font=('Ariel', 38, "bold"), bg="#282828", fg="whitesmoke")
header2.pack(padx=5)

# Button to switch no choose document window
# Create a custom style with a rounded border radius
style = ttk.Style()
style.configure("RoundedButton.TButton", relief="flat", font=('Arial', 18, 'bold'), background="#00bcd4",
                foreground="whitesmoke", weight=root.winfo_screenwidth(),
                justify="center", bd=0, highlightthickness=2, highlightbackground="#00bcd4")
style.layout("RoundedButton.TButton",
             [('Button.focus', {'children': [('Button.label', {'sticky': 'nswe'})]})])
btn_txt = "Choose\nDocument"
btn_choose_doc = ttk.Button(root, text=btn_txt, command=show_choose_doc_screen, style="RoundedButton.TButton")
btn_choose_doc.pack(pady=50)

root.mainloop()
