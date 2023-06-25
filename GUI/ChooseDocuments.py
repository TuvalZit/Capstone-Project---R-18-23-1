# Imports
from tkinter import ANCHOR
from Common import *
from tkinter import ttk
from Results import displayResults

###################################################################
directory = 'data_base/imposters'

###################################################################
class ChooseDocScreen:
    def __init__(self):
        self.root = tk.Tk()
        setWindowStyle(self.root, "PART : Pre-trained Authorship Embedding Transformer>Choosing Document")
        header = tk.Label(self.root, text="Choose document:", font=('Ariel', 38), bg="#282828", fg="whitesmoke")
        header.grid(row=0, column=0, sticky="n", pady=10)

        textsList = []
        extractTextList(directory, textsList)
        # Create a Frame for the Listbox and Scrollbar
        frame = tk.Frame(self.root)
        frame.configure(relief="solid", highlightbackground="#00bcd4", highlightthickness=1)
        frame.grid(row=1, column=0, padx=10, pady=10)

        self.choiseBox = tk.Listbox(frame, width=120, height=15, selectmode='SINGLE', selectbackground="#00bcd4")
        self.choiseBox.pack(side="left", fill="both")

        def on_hover(event):
            # Retrieve the index of the item under the mouse cursor
            index = self.choiseBox.nearest(event.y)

            for i in range(self.choiseBox.size()):
                self.choiseBox.itemconfig(i, background="")
            # Activate (select) the item under the mouse cursor
            self.choiseBox.activate(index)
            # Clear all tags from the Listbox items
            # Apply the hover effect to the item under the mouse cursor
            self.choiseBox.itemconfig(index, background="#00bcd4")

        # Function to handle selection in the Listbox
        def handle_selection():
            try:
                # Get the selected item from the Listbox
                selected_item = self.choiseBox.get(tk.ACTIVE)
            except tk.TclError:
                show_custom_messagebox("Error", "No item selected.")

        self.choiseBox.bind("<Motion>", on_hover)

        scrollbar = tk.Scrollbar(frame, command=self.choiseBox.yview)
        scrollbar.pack(side="right", fill="y")

        self.choiseBox.config(yscrollcommand=scrollbar.set)
        self.choiseBox.config(yscrollcommand=lambda *args: scrollbar.set(*args))
        # Insert elements into the listbox
        for txt in textsList:
            self.choiseBox.insert(tk.END, txt.split(".")[0])

        style = ttk.Style()
        style.configure("RoundedButton.TButton", relief="flat", font=('Arial', 18, "bold"), background="#00bcd4",
                        foreground="whitesmoke", justify="center", bd=0, highlightthickness=2,
                        highlightbackground="#00bcd4")
        style.layout("RoundedButton.TButton",
                     [('Button.focus', {'children': [('Button.label', {'sticky': 'nswe'})]})])

        btn_continue = ttk.Button(self.root, text="Create Embedding", command=self.handleContinue,
                                  style="RoundedButton.TButton")
        btn_continue.grid(row=2, column=0, pady=2)

        num_rows = self.root.grid_size()[1]
        for row in range(num_rows):
            self.root.grid_rowconfigure(row, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.mainloop()

    # Handle what happens when click the continue button
    def handleContinue(self):
        selected_item = self.choiseBox.get(ANCHOR)
        if selected_item:
            self.root.destroy()
            screen = displayResults(selected_item)
        else:
            show_custom_messagebox("No Selection", "Please select an option from the Listbox.", self.root)
